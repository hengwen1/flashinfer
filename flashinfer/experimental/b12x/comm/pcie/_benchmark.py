"""Private operands and arithmetic checks for prepared PCIe tuning races."""

from functools import partial

import torch

from b12x.preparation import PreparedCall


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class _ReductionInputs:
    def __init__(self, state, *, inplace):
        self.state, self.step = state, 0
        query = state.query
        call = query.call
        self.twoshot = query.surface.startswith("PCIeTwoShotBF16.")
        self.fused = not query.surface.endswith(".all_reduce")
        shape = tuple(call["shape"] if self.twoshot else query.setup["shape"])
        dtype = _DTYPES[call["dtype"]]
        device = state.runtime.device
        generator = torch.Generator().manual_seed(81721)
        # Exactly representable partial sums isolate protocol and binding errors.
        values = (
            torch.randint(
                -16, 17, (query.world_size, *shape), generator=generator
            ).float()
            / 32
        )
        self.expected_sources = (values, torch.zeros_like(values), -values)
        self.seeds = tuple(
            value[query.rank].to(device=device, dtype=dtype)
            for value in self.expected_sources
        )
        self.source = torch.empty(shape, dtype=dtype, device=device)
        self.out = self.source if inplace else torch.empty_like(self.source)
        self.residual = self.weight = self.addend = self.group_weights = (
            self.group_ids
        ) = None
        self.residual_seed = None
        if self.fused:
            logical = tuple(query.setup["logical_shape"])
            self.residual_seed = torch.full(
                logical, 0.25, dtype=_DTYPES[call["residual_dtype"]], device=device
            )
            self.residual = torch.empty_like(self.residual_seed)
            self.out = (
                torch.empty(logical, dtype=_DTYPES[call["output_dtype"]], device=device)
                if call["normalize"]
                else None
            )
            self.weight = (
                torch.ones(
                    logical[-1], dtype=_DTYPES[call["weight_dtype"]], device=device
                )
                if call["normalize"]
                else None
            )
            if call["input_addend"]:
                self.addend = torch.full(logical, 0.125, dtype=dtype, device=device)
            if call["group_size"]:
                group_shape = shape[:2]
                self.group_weights = torch.full(
                    group_shape, 0.5, dtype=torch.float32, device=device
                )
                self.group_ids = torch.zeros(
                    group_shape,
                    dtype=getattr(torch, call["group_ids_dtype"]),
                    device=device,
                )

    def produce(self, index):
        self.step = index
        self.source.copy_(self.seeds[index])
        if self.residual is not None:
            self.residual.copy_(self.residual_seed)

    def run(self):
        state, call = self.state, self.state.query.call
        if self.twoshot:
            return state.run(
                self.source,
                None,
                self.out,
                threads=call["threads"],
                block_limit=call["block_limit"],
            )
        if not self.fused:
            state.run_plain(self.source, self.out)
            return self.out
        state.run_fused(
            self.source,
            self.residual,
            self.weight,
            self.out,
            self.residual,
            1e-6,
            self.addend,
            self.group_weights,
            self.group_ids,
            call["group_id_limit"],
        )
        return self.residual if self.out is None else self.out

    def validate(self):
        query, call = self.state.query, self.state.query.call
        values = self.expected_sources[self.step]
        dtype = self.source.dtype
        if self.fused:
            if call["group_size"]:
                values = (values * 0.5).sum(dim=2)
                if call["group_id_limit"] == 0:
                    values.zero_()
                values = values.to(dtype).float()
            if call["input_addend"]:
                values = (values + 0.125).to(dtype).float()
        expected = values.sum(dim=0)
        if not self.fused:
            torch.testing.assert_close(
                self.out.cpu(), expected.to(dtype), rtol=0, atol=0
            )
            return
        if call["reduction_dtype"] is not None:
            expected = expected.to(_DTYPES[call["reduction_dtype"]]).float()
        expected = expected + 0.25
        torch.testing.assert_close(
            self.residual.cpu(), expected.to(self.residual.dtype), rtol=0, atol=0
        )
        if self.out is not None:
            expected = expected * torch.rsqrt(
                expected.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            if call["zero_centered"]:
                expected *= 2
            tolerance = (
                8e-3
                if self.out.dtype == torch.bfloat16
                else 2e-3
                if self.out.dtype == torch.float16
                else 2e-5
            )
            torch.testing.assert_close(
                self.out.cpu(),
                expected.to(self.out.dtype),
                rtol=tolerance,
                atol=tolerance,
            )


def benchmark_call(state, *, inplace=False):
    dcp = state.query.surface.startswith(("DcpAllToAll.", "DcpAllToAllPool."))
    inputs = _DcpInputs(state) if dcp else _ReductionInputs(state, inplace=inplace)
    if not dcp and not inputs.twoshot:
        state.bind_input(inputs.source)
    # Callbacks own the operands; the operands must not own callbacks bound to
    # themselves, or GPU tensors survive until an unrelated cyclic collection.
    producers = tuple(
        partial(inputs.produce, index) for index in range(len(inputs.seeds))
    )
    return PreparedCall(
        run=inputs.run,
        output=inputs.out,
        produce=producers[0],
        benchmark_producers=producers,
        validate=inputs.validate,
        owners=(inputs, state.runtime),
        capture_context=None if dcp else state.runtime.capture,
    )


class _DcpInputs:
    def __init__(self, state):
        self.state, self.step = state, 0
        query, runtime = state.query, state.runtime
        call = query.call
        self.lse_reduce = query.surface.endswith(".lse_reduce_scatter")
        shape = tuple(call["shape"])
        name = call["dtype"].removeprefix("torch.")
        dtype = _DTYPES.get(name) or getattr(torch, name)
        generator = torch.Generator().manual_seed(9419)
        stacked = torch.randint(
            0, 16, (query.world_size, *shape), generator=generator
        ).to(dtype)
        self.seeds = tuple(
            value[query.rank].to(runtime.device)
            for value in (stacked, torch.zeros_like(stacked))
        )
        self.source = torch.empty_strided(
            shape, tuple(call["stride"]), dtype=dtype, device=runtime.device
        )
        self.out = torch.empty_strided(
            tuple(call["out_shape"]),
            tuple(call["out_stride"]),
            dtype=dtype,
            device=runtime.device,
        )
        self.lse = None
        if self.lse_reduce:
            from .pcie_dcp_a2a import lse_reduce_scatter_reference

            scores = (
                torch.arange(query.world_size, dtype=torch.float32)[:, None, None]
                .expand(query.world_size, *shape[:2])
                .contiguous()
                / 8
            )
            self.lse = torch.empty_strided(
                tuple(call["lse_shape"]),
                tuple(call["lse_stride"]),
                dtype=torch.float32,
                device=runtime.device,
            )
            self.lse.copy_(scores[query.rank])
            expected = lse_reduce_scatter_reference(
                stacked, scores, query.rank, is_lse_base_on_e=call["natural_log"]
            )
        else:
            expected = stacked.permute(1, 0, 2, 3).reshape(self.out.shape)
        self.expected = (expected, torch.zeros_like(expected))

    def produce(self, index):
        self.step = index
        self.source.copy_(self.seeds[index])

    def run(self):
        if self.lse_reduce:
            return self.state.run(
                partial_output=self.source,
                partial_lse=self.lse,
                out=self.out,
                is_lse_base_on_e=self.state.query.call["natural_log"],
            )
        return self.state.run(local_input=self.source, out=self.out)

    def validate(self):
        tolerance = (
            (2e-2 if self.out.dtype == torch.bfloat16 else 2e-3)
            if self.lse_reduce
            else 0
        )
        torch.testing.assert_close(
            self.out.cpu(), self.expected[self.step], rtol=tolerance, atol=tolerance
        )
