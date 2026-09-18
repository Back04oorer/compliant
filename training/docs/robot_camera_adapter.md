# Robot and camera adapter checklist

## Robot

- Map measured joint positions/efforts into the checkpoint's named state order.
- Map canonical single-arm actions back to the selected physical arm.
- Confirm radians/metres and the gripper open/closed convention.
- Define startup/reset behavior independently of the learned policy.
- Enforce position, per-step, velocity, acceleration, stale-data, and emergency
  stop guards after postprocessing and before the command mux.

## Cameras and stereo

- Recalibrate stereo intrinsics/extrinsics for the target hardware.
- Rectify with the same crop and output size used during conversion.
- Keep left/right stereo roles fixed and verify depth orientation.
- Match RGB exposure/white balance as closely as practical, but do not use
  color postprocessing to conceal a viewpoint/calibration mismatch.
- Match every live frame to the nearest measured state using timestamps and
  reject observations beyond the permitted synchronization lag.
- FastFS settings are part of the checkpoint contract, not cosmetic runtime
  tuning.

## Dataset

- The model never predicts future observations. `n_obs_steps=1` uses the
  current synchronized observation; larger observation chunks use past and
  current frames only.
- Labels must represent the action intended for the observation timeline after
  command/measurement alignment.
- Inspect converted videos, state/action plots, gripper events, synchronization
  lag, and episode boundaries before training.

See [adapting_to_your_robot.md](adapting_to_your_robot.md) for the full
conversion-topic and feature-selection discussion.
