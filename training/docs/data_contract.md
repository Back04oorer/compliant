# Data contract

Stereo-ACT consumes a local LeRobot v3 dataset. Images are RGB tensors and
FastFS depth is metric depth in meters, stored as a frame-indexed mmap sidecar.

## Bimanual

- `observation.state [28]`: left position 7, left raw effort 7, right position
  7, right raw effort 7.
- `action [14]`: left joints/gripper 7, right joints/gripper 7.
- Visual keys: top RGB, left wrist RGB, right wrist RGB, FastFS depth.

## Single active arm

- `observation.state [14]`: active-arm position 7 and raw effort 7.
- `action [7]`: active-arm joints/gripper.
- Visual keys: top RGB, canonical right wrist RGB, FastFS depth.

`build_active_arm_lerobot_subset.py` maps either physical left or physical
right demonstrations into the same canonical policy slot. This permits a
single checkpoint to be trained from both sides after separately canonicalized
datasets are merged. Mapping policy output back to a physical robot is a
deployment responsibility and is intentionally robot-specific.

## Variant inputs

| Variant | Top DINO | FastFS + DeFM | Wrist encoder | Fusion |
|---|---:|---:|---|---|
| Fused | yes | yes | ResNet18 | DINO Q ← DeFM K/V + 2D relative bias |
| DINO-only | yes | no | ResNet18 | direct |
| DeFM-only | no | yes | ResNet18 | direct |

“DINO-only” refers to the top-camera encoder. Wrist cameras use ResNet18 in all
three standard comparisons.
