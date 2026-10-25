"""Full-width validation for caller-supplied expert IDs."""

import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32


@cute.jit
def local_expert_id(raw_expert, num_experts: Int32):
    """Return an addressable local expert or the inactive-route sentinel."""
    expert = Int32(-1)
    if raw_expert >= 0 and raw_expert < num_experts:
        expert = raw_expert.to(Int32)
    return expert
