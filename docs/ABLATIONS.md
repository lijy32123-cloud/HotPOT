# Ablation design

## Shared representation

The first pair compares Full against Untied. Untied gives each family its own history encoder while retaining the full route's encoder capacity. This tests whether using common history-encoding weights provides a useful sharing boundary and reduces parameter storage.
The second pair moves the shared trainable representation later, after PDE-specific field coupling. Late and Late-Untied use the same fixed history map and later block capacity; Late-Untied makes one copy of that block per family. These comparisons test the placement and effect of sharing, rather than treating every extra parameter as an architectural advantage.

## Numerical experts

Direct replaces the numerical program with direct next-state prediction. The architecture, the stored parameters and the training protocol are unchanged, so the comparison tests the complete predictor–update construction at equal capacity.

Generic iteration retains repeated predictor queries on the NS route but replaces the equation-specific update with a generic residual iteration. Comparing it with Full addresses whether the gain can be explained only by repeated refinement. Other PDE routes retain their full execution.

All public builders emit the same six retained history features into the checkpoint-compatible input layout. The unused layout slots are zero before any learned history map. No variant in this release enables the discarded history-difference channels.

Run ablations with the 17M setting, the same source seed, row counts, update budget, and evaluation protocol. The CLI values are `full`, `untied`, `late`, `late_untied`, `direct`, and `generic_iteration`. Use `model.parameter_ledger()` to inspect stored and active parameter counts.
