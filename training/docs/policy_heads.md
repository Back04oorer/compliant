# Policy heads

All heads consume the same fused observation memory; changing a head does not
change DINO/DeFM semantics, wrist-camera ordering, or the state contract.

## Stereo-ACT

The standard model uses a 4-layer ACT encoder, 7-layer decoder, and 50-action
joint-position chunk. Training uses the ACT-HF processor stored with the
checkpoint. Deployment may queue a chunk or replan every observation and
temporally ensemble chunks aligned by execution timestamp.

## Stereo-DP

The policy uses a compact ConditionalUnet1D (`128,256,512`), a 32-step action
horizon, 100 training diffusion timesteps, and random single-timestep noise
prediction during training. Inference integrates the same learned denoising
field with configurable DDIM steps (commonly 4 or 16). The number of queued
actions is independent of DDIM iterations. Temporal ensembling is an executor
feature: align overlapping chunks by their intended execution timestamps before
combining them.

## Stereo Flow Matching V2

V2 keeps the full Stereo-ACT condition memory, passes it through a 4×512 ACT
condition encoder, and predicts a continuous flow with a 4×256 conditional flow
decoder. The standard horizon is 32, execute length 20, and integration count
10. Its RTC path accepts the unexecuted prefix of the previous chunk plus the
measured inference delay; therefore RTC requires an asynchronous executor and
must call `predict_action_chunk`, not queue-mode `select_action`.

The legacy `stereo_flow` module remains because V2 inherits its visual/config
contract. It is not exposed as a primary training choice.
