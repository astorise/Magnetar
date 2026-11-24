## Decisions

### Forbid segment starts, not move KV

The multi-GPU Provider copies any input another GPU produced, so a split pair would still compute correctly. The planner avoids the split instead, because the source's KV history grows at every step, and every copy would carry the whole history.

For a pair `(layer, source)`, no segment may start at any layer in `source + 1..=layer`. Gemma 4 E2B's last 20 layers reuse layers 13 and 14, so its segments can only split at or before layer 13.

### Grow at the target, shrink at the budget

- **Balanced target:** a balanced share is a soft target, so a segment that reaches it at a joined layer keeps growing to the next allowed start.
- **Budget:** a Device's budget is hard, so a segment that would exceed it ends at the last allowed start before that point.
- **No fit:** joined layers that no Device can hold together do not fit, and the model is refused, as for any placement that does not fit.

## Evidence

- `layers_sharing_kv_stay_with_their_source`:
  - **Balanced split:** a split that would fall between joined layers moves past them.
  - **Budget:** a tight budget ends the segment before the source, where the unconstrained placement splits at layer 4.
  - **No fit:** joined layers too large for every Device do not fit.
- `layer_placement_keeps_kv_sharing_layers_with_their_source`: the production placement marks the layers from the artifact's layout.
- **Not run:** a placement on two real GPUs.
