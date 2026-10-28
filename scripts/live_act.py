"""Stepwise live ACT state: latest target, local pause/reset, and unchanged motion guards."""
from rgb_act_execution import ExecutionGuard, finite_vector


class LiveACTGuard(ExecutionGuard):
    def __init__(self, config, reset_positions=None):
        super().__init__(config, reset_positions, mit_j2_j3=True)
        self.continuous = True
        self.policy_epoch = 0
        self.target = self.previous_target = None
        self.pause_started = self.resume_phase = None
        self.hold_after_reset = False

    def arm(self, measured, diagnostics, now):
        super().arm(measured, diagnostics, now)
        self.policy_epoch += 1
        self.target = self.previous_target = None
        self.hold_after_reset = False

    def validate_actions(self, actions, measured):
        if not isinstance(actions, list) or len(actions) != 1:
            raise ValueError('Live ACT requires exactly one action per fresh observation')
        target = finite_vector(actions[0], 14, 'live ACT action')
        low, high = self.config['arm_position_min'][6], self.config['arm_position_max'][6]
        # Saturate model gripper targets; feedback, reset poses and joint limits stay strict.
        for index in (6, 13):
            target[index] = max(low, min(high, target[index]))
        target = self.position(target)
        measured = self.measured_position(measured)
        previous = self.previous_target
        if previous is None:
            previous = self.last_command if self.last_command is not None else measured
        # TicTac medium guard: limit target increments, then EMA; output slew/tracking checks still apply.
        increments = self.limits(.05, .005)
        guarded = [old + .4 * max(-limit, min(limit, goal-old))
                   for goal,old,limit in zip(target,previous,increments)]
        return [self.position(guarded)]

    def submit(self, payload, measured, ros_now, now):
        if self.phase != 'policy' or payload.get('policy_epoch') != self.policy_epoch:
            raise ValueError('Live ACT requires the current policy epoch and running phase')
        # A newer observation replaces the pending target; no future action chunk is replayed.
        self.queue.clear()
        super().submit(dict(payload, actions=[payload['action']]), measured, ros_now, now)
        self.target = self.previous_target = self.queue.popleft()
        self.reason = 'ARMED: live ACT step %d' % self.sequence

    def toggle_pause(self, measured, now):
        if not self.active:
            raise ValueError('Type ARM + Enter before using ACT controls')
        if self.phase == 'paused':
            self.phase = self.resume_phase
            self.started += now - self.pause_started
            self.last_plan = now
            self.reason = 'Live ACT resumed; waiting for fresh observations'
        else:
            self.resume_phase, self.phase = self.phase, 'paused'
            self.pause_started = now
            self.reason = 'Live ACT paused; holding measured pose'
        self.last_command = self.measured_position(measured)
        self.velocity = [0.] * 14
        self.target = self.previous_target = None
        self.queue.clear()
        self.reset_settled_since = None
        self.policy_epoch += 1

    def rewind(self, measured, now):
        if not self.active or self.reset_positions is None:
            raise ValueError('Returning to first frame requires local ARM and a dataset reset pose')
        self.last_command = self.measured_position(measured)
        self.velocity = [0.] * 14
        self.target = self.previous_target = None
        self.queue.clear()
        self.phase, self.reason = 'reset', 'Returning slowly to dataset first frame; Space resumes after positioning'
        self.started = self.last_tick = self.last_plan = now
        self.reset_settled_since = None
        self.hold_after_reset = True
        self.policy_epoch += 1

    def step(self, measured, now, stationary=True):
        if not self.active:
            return None
        previous_phase = self.phase
        if self.phase == 'policy' and now - self.last_plan > self.config['plan_timeout_s']:
            raise ValueError('Live ACT plan watchdog expired')
        if self.phase == 'policy' and self.target is not None:
            self.queue.append(self.target)
        command = super().step(measured, now, stationary)
        if previous_phase == 'reset' and self.phase == 'policy' and self.hold_after_reset:
            self.hold_after_reset = False
            self.phase, self.resume_phase = 'paused', 'policy'
            self.pause_started = now
            self.reason = 'At dataset first frame; Space resumes live ACT'
            self.policy_epoch += 1
        return command
