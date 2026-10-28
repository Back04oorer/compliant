# RGB ACT execution through the shared follower mux

TUI `m` robot/dry-run modes use live RGB ACT predictions with the same follower
launch as teleoperation and rosbag replay. A command mux switches between leader
feedback and the shared policy/replay input; drivers stay running across modes. The model receives three
live RGB images and a 28-value state, then calls `select_action()` for one action per new observation. The ACT network
still predicts a 50-step horizon internally; temporal ensemble (coefficient 0.01)
fuses overlapping predictions. `n_action_steps=1` matches TicTac's default ACT
deployment. The bridge consumes only the selected 14-value action, not a future
action queue. The policy loop targets 30 Hz; actual updates depend on fresh camera
observations and measured inference latency. The bridge publishes at 30 Hz while
tracking the latest accepted target with speed/acceleration limits.
Prediction-only mode (`m` → `predictions`) creates no execution bridge or actuator publishers.
The mux ACT path has software tests; real motion requires local ARM.

## Start from the TUI

Checkpoint selection supports a candidate list, `/` filtering, directory browsing
and Tab for a pasted path. Mode/device use arrow-key lists. Invalid input stays
on the current field; Esc returns without starting a trial. See
[shared TUI controls](01_startup.md#tui-selection-and-input).

Run the TUI inside `piper-control`:

```bash
docker exec -it -w /opt/piper piper-control python3 scripts/teleop_tui.py --execute
```

1. Exit any previous bag replay or ACT bridge so only one policy/replay publisher
   owns the policy input. Keep the existing follower drivers running.
2. If followers are not running, choose `2` → `1` in another terminal. This shared
   launch enables the followers, starts follower gravity and the mux, and defaults
   to teleop input. Use this same driver terminal for teleop, ACT and bag replay.
   Leaders (`2` → `2`) are needed only for teleoperation. Menu 9 has been removed.
3. Camera feeds start automatically when you enter `m`. Existing feeds are
   reused; only missing roles are started. Roles/devices and output size are saved
   in `outputs/tui/inference_cameras.json`. If no mapping is saved and no complete
   known camera setup is live, configure Left wrist / Right wrist / Top once with
   the [standalone camera setup command](07_rgb_act_predictions.md#live-observations-still-predictions-only);
   subsequent inference runs reuse that mapping. The follower drivers provide
   the required measured state.
4. Choose `m`, accept the latest complete checkpoint or select another, then choose
   `robot` or `dry-run` and `cuda`. The launcher loads the model and starts its
   private read-only observer and execution bridge.
5. Wait for `Bridge ready`, then type `ARM` and Enter in the same terminal. After
   feedback/connection checks, it seeds a measured-pose hold, selects the policy
   input on both muxes and confirms selection before motion. It first positions slowly to dataset episode 0, frame 0 measured pose, then runs live ACT
   continuously. Space pauses/resumes while holding the measured pose. Enter
   returns slowly to the dataset first frame and holds there; Space resumes
   inference after positioning. Pause/reset keep the mux in policy mode. `q` or
   Ctrl-C clears ACT targets, returns slowly to the dataset first frame, then
   exits and restores teleop input. Align the leaders before handing
   control back: live leader positions become follower targets after restoration.
   Stop the previous bag replay before running ACT: both require exclusive policy
   command publishers.

The driver startup has a CLI equivalent:

```bash
docker exec -it -w /opt/piper piper-control python3 scripts/piper.py robot --execute
```

No driver restart or manual policy selection is needed. Local ARM switches the
mux; stop/exit restores teleop. Menu `2` → `7` provides manual teleop takeover,
which stops ACT commands; `2` → `8` selects the policy input manually. Neither
manual selection starts or arms ACT. TUI preview mode only prints commands.

## Position rules and motion guards

Live ACT uses the recorded position-only MIT convention for **J2 and J3 on both
arms**: values must be finite and within the MIT protocol input range
`[-12.5, 12.5]` radians. Negative J2 and positive J3 are preserved in reset poses,
measured feedback, and every model target. There is no zero offset, sign change,
or clipping of those targets. This protocol range is not a mechanical travel or
collision limit. Other joint and gripper position bounds remain those in
`config/rgb_act_execution.json`, including J5 ±70 degrees and gripper 0..0.08 m.
Gripper endpoint roundoff of at most `1e-8` m is normalized to the configured
bound before further checks. Finite live ACT model gripper targets are clipped to
the configured range (normally `0..0.08` m) before target smoothing, regardless of
the overshoot size. NaN/Inf still stop ACT. Clipping applies only to model gripper
targets, not measured feedback, dataset reset poses, or arm joints. Rejection
messages show 12 significant digits and the excess beyond the bound.
The baseline guard/legacy bridge retain their original bounds unless live ACT
is explicitly selected by the launcher or `--live-act` on the bridge.

The first-frame pose comes from the checkpoint dataset's `observation.state`
(left positions 0:7 and right positions 14:21), not its efforts or recorded action.
Positioning uses the same speed and acceleration shaping as ACT, with a 60-second
positioning timeout. It requires stationary joint feedback, slow commands and
joint errors <=0.05 rad / gripper errors <=0.002 m for 0.5 seconds. The joint
threshold matches the recording-compatible MIT reset and accommodates its static
following error. After positioning, all five observation timestamps must be newer
than completion before the model can submit an action. Live ACT has no five-second
or sixty-second rollout cutoff. The legacy `trial_seconds` config field remains
for chunk/replay compatibility and does not limit continuous live ACT; the reset stage
still has its sixty-second timeout.

Each selected action must pass finite-number, shape/order/rate, fresh synchronized
observation, new timestamp and position checks. Valid raw goals use the TicTac
medium target guard: clamp each update to ±0.05 rad per joint / ±0.005 m per
gripper relative to the previous guarded target, then apply EMA alpha 0.4. The
first target starts from the current commanded/measured pose. The legacy 0.15-rad
initial-distance rejection and raw adjacent-target jump rejection do not apply to
this continuous live path. Joint position violations and non-finite values still
fail before shaping. The bridge retains 0.2 rad/s joint speed, 1 rad/s² joint acceleration,
0.015 m/s gripper speed and 0.05 m/s² gripper acceleration, plus tracking,
CAN-feedback, control-loop, plan and client-heartbeat watchdogs. There is no wait
for a 50-point queue to drain. Between model updates the controller approaches the
latest target; the target still expires after the plan timeout if no new prediction
arrives. Slow inference is not treated as permission to reuse stale observations.
The client reports actual accepted steps per second. Each accepted step event
records `raw_action` (including any gripper overshoot) and `guarded_target` so
model output can be distinguished from clipping, target shaping and the final
speed-limited commands.

Pause and reset invalidate the pending target and increment a policy epoch. The
client calls `policy.reset()` to clear temporal ensemble on each epoch change.
The `/step` endpoint discards predictions from an earlier epoch or a paused/reset
phase without resuming or disarming the session. Thus a prediction already in
flight when Space/Enter is pressed cannot restore an old command. Resumption
requires observations newer than the key action. Returning to the first frame
clears both the target and policy cache, and waits for Space before restarting.

Dry-run uses the same live logic without actuator publishers. It reads actual
feedback, so it cannot simulate successful positioning from a different pose or
prove that a changing policy target tracks when the physical robot stays still.
Prediction mode remains chunk-based and never moves the robot. For a
no-positioning live trial, invoke `test_rgb_act.py` without `--reset-to-first-frame`;
it validates one selected action before starting the bridge and resets the policy
cache afterwards. TUI `m` always includes first-frame positioning.

## Mux routing and driver checks

Only `sensor_msgs/JointState.position` is published to:

- `/robot/arm_left/vla_joint_cmd`
- `/robot/arm_right/vla_joint_cmd`

The same policy input topics carry ACT or bag replay, one at a time. Each mux
selects either its leader's `/teleop/arm_*/joint_states_single` or the policy input,
then publishes `/robot/arm_*/joint_cmd_mux`. Both joint and gripper driver
subscriptions must point to this mux output. Joint order is joint1..joint6, then
gripper; units are radians and metres. No velocity or effort command is sent.
Follower gravity remains owned by the existing driver.

Robot mode requires one exclusive policy publisher and one mux subscriber per
arm, the expected mux output/status publishers, a driver subscribed to each mux
output, and position-only MIT parameters (`ctrl_mode=mit`, `enable_pos=true`,
`enable_vel=false`, `enable_tor=false`). The bridge confirms both selections on
ARM and continuously verifies that policy remains selected while armed, including
pause/reset. Manual takeover, a conflicting publisher, missing connections, stale
CAN feedback, driver faults or loss of required gravity stop execution. A cold mux
may initialize its output only after local ARM; until then only measured holds
are sent, with a one-second initialization deadline. Partial handoff failure
rolls attempted arms back to teleop; it does not leave the trial armed.

Disarmed readiness accepts the normal teleop selection. Dry-run never publishes
actuator commands or switches the mux, and can observe disabled drivers. The
bridge cannot start drivers, change calibration/parameters or enable motors.
The only control service it calls is the existing mux selection service.

The camera supervisor starts before the observer and saves its ownership record
in `camera_session.json`, with camera logs in `cameras.log`. On exit, only cameras
started by this inference session are stopped; reused camera processes are left
running. An explicit external `--observer-url` keeps its camera lifecycle external.
Missing/stale frames still block the observer and execution. The camera supervisor
never starts robot drivers or sends robot commands.

The model runs in `/opt/piper-inference`; ROS children use `/usr/bin/python3` with
a rebuilt ROS environment. Each trial has a loopback-only HTTP endpoint, random
authentication token and private 0600 session file. ARM is available only in the
local bridge terminal. The launcher checks the bridge's `routing=mux` contract, `live_act` flag, mode,
reset target, observation mode and limits before connecting the client.

Outputs are saved under `outputs/model_tests/<mode>_<timestamp>`:
`reset_pose.json`, `execution_config.json`, `contract.json`, `preflight.json` (including continuous live mode) and
`bridge/` events/session. Reports identify live ACT and the J2/J3 protocol bounds.
The no-positioning workflow also saves `preflight_prediction.npy`. The managed
observer uses a dynamically assigned port and is stopped on exit; an observer
supplied with `--observer-url` is reused and left running.

## Manual bridge and stopping

The TUI handles bridge/client startup. For manual operation, add `--live-act`
to `rgb_act_ros_executor.py` (without `--states-only` or `--recording-plan`). The
standalone client `execute_rgb_act.py` accepts the resulting session; real output
requires `--execute` on the bridge and `--allow-execution` on that client, followed
by local ARM. Reuse of an existing session directory is rejected.

For an armed live ACT session with a dataset start pose, `q` or the first Ctrl-C
returns slowly to that pose before disarming and exiting. The return uses the
same speed/acceleration shaping, feedback checks and 60-second positioning limit
as Enter. The bridge owns this return while the launcher keeps cameras and the
observer running; no model predictions are accepted during it. Before ARM, exit
does not move the arms. Predictions mode remains read-only; dry-run publishes no
actuator commands and cannot simulate a physical return from another pose.

A second Ctrl-C aborts the return immediately. If it was started with `q`, press
Ctrl-C twice to abort. Faults, lost feedback, manual takeover and timeouts stop
immediately rather than attempting a return. Space pauses at the measured pose;
Enter returns to the first frame and holds. On stopping, it attempts a measured-pose hold only
while feedback and exclusive mux connections remain valid, then restores teleop
for arms still owned by the bridge. It respects manual takeover to another input.
Failed restoration is logged and reported with the manual recovery entry `2` → `7`.
A fault or timeout does not auto-resume. Motor power remains unchanged. A killed bridge cannot send a
hold or restore the mux; the driver retains its last target when its publisher
disappears. Use menu `2` → `7` for manual teleop takeover if needed. This
software does not provide collision planning, scene restoration or a hardware
emergency stop. Offline tests do not establish physical stopping or task success.
