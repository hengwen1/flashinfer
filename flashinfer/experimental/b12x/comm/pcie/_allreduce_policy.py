"""Preparation-time crossovers for the TP4 BF16 PCIe transports."""

# Graph replay measurements on four SM120 GPUs with 188 SMs per device.
TP4_ONESHOT_MAX_BYTES = 24 * 1024
TP4_RSAG_MAX_BYTES = 640 * 1024
BF16_TWOSHOT_MIN_BYTES = 64 * 1024
BF16_TWOSHOT_MAX_BYTES = 2 * 1024 * 1024


def tp4_bf16_transport(
    nbytes: int, *, payload_ready: bool, twoshot_available: bool
) -> str:
    """Choose among transports whose storage and alias contracts are satisfied."""
    if payload_ready:
        if nbytes <= TP4_ONESHOT_MAX_BYTES:
            return "tp4_push_sentinel"
        if twoshot_available and nbytes > TP4_RSAG_MAX_BYTES:
            return "twoshot"
        return "tp4_push_rsag"
    if twoshot_available and nbytes >= BF16_TWOSHOT_MIN_BYTES:
        return "twoshot"
    return "pull"
