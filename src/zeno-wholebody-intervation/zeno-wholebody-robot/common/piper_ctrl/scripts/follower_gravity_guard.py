#!/usr/bin/env python3
"""Pure, clock-injected safety filter for follower gravity (no ROS or CAN)."""
import math
import threading

NAMES = ['joint%d' % i for i in range(1, 7)]


def six_positive(values, label):
    result = [float(x) for x in values]
    if len(result) != 6 or not all(math.isfinite(x) and x > 0 for x in result):
        raise ValueError(label + ' requires six finite positive values')
    return result


class FeedforwardGuard:
    def __init__(self, limits, slew, timeout=0.10, ramp_sec=0.5):
        self.limits = six_positive(limits, 'limits')
        self.slew = six_positive(slew, 'slew')
        self.timeout = float(timeout)
        self.ramp_sec = float(ramp_sec)
        if max(self.limits) > 18:
            raise ValueError('gravity limits must not exceed SDK t_ref range (+/-18)')
        if not all(math.isfinite(x) and x > 0 for x in (self.timeout, self.ramp_sec)):
            raise ValueError('timeout/ramp must be finite and positive')
        self.lock = threading.Lock()
        self.sample = None
        self.last_stamp = None
        self.last_advance = None
        self.output = [0.] * 6
        self.ramp_from = [0.] * 6
        self.valid_since = None
        self.last_tick = None

    def invalidate(self):
        with self.lock:
            self.sample = None
            self.valid_since = None

    def receive(self, names, effort, stamp, ros_now, now):
        """stamp is the SOURCE STATE time, not calculator publication time.

        Repeated source stamps can update a ramped target but cannot refresh
        the monotonic source watchdog, including when ROS time is frozen.
        """
        try:
            if len(names) != 6 or set(names) != set(NAMES) or len(effort) != 6:
                raise ValueError('expected six uniquely named gravity efforts')
            values = dict(zip(names, effort))
            target = [float(values[n]) for n in NAMES]
            if not all(math.isfinite(x) for x in target + [stamp, ros_now, now]):
                raise ValueError('nonfinite gravity sample/time')
            if stamp <= 0 or not -0.02 <= ros_now - stamp <= self.timeout:
                raise ValueError('stale/future gravity source state')
            with self.lock:
                if self.last_stamp is not None and stamp < self.last_stamp:
                    raise ValueError('out-of-order gravity source state')
                if self.last_advance is None or now - self.last_advance > self.timeout:
                    self.valid_since = None
                if self.last_stamp is None or stamp > self.last_stamp:
                    self.last_stamp = stamp
                    self.last_advance = now
                self.sample = (target, stamp)
            return True, ''
        except (ValueError, TypeError, KeyError) as exc:
            self.invalidate()
            return False, str(exc)

    def step(self, now, ros_now, active=True, feedback_ok=True):
        """Return (bounded t_ref, ready, reason); only control thread calls it."""
        with self.lock:
            gap = 0. if self.last_tick is None else max(0., now - self.last_tick)
            dt = min(gap, 0.05)  # scheduling stalls never permit large torque jumps
            self.last_tick = now
            if not active:
                self.output = [0.] * 6
                self.valid_since = None
                return self.output[:], False, 'inactive'
            if gap > self.timeout:
                self.valid_since = None
            reason = 'ready'
            if not feedback_ok:
                reason = 'CAN feedback not advancing'
            elif self.sample is None or self.last_advance is None:
                reason = 'missing/invalid gravity'
            elif not 0 <= now - self.last_advance <= self.timeout:
                reason = 'gravity source stalled (wall clock)'
            elif not -0.02 <= ros_now - self.sample[1] <= self.timeout:
                reason = 'gravity source stale (ROS clock)'
            valid = reason == 'ready'
            target = [0.] * 6
            ready = False
            if valid:
                if self.valid_since is None:
                    self.valid_since = now
                    self.ramp_from = self.output[:]
                alpha = min(max((now - self.valid_since) / self.ramp_sec, 0.), 1.)
                for i, raw in enumerate(self.sample[0]):
                    clipped = max(-self.limits[i], min(self.limits[i], raw))
                    target[i] = self.ramp_from[i] + alpha * (clipped - self.ramp_from[i])
                ready = alpha >= 1.
                if not ready:
                    reason = 'ramping'
            else:
                self.valid_since = None
            for i in range(6):
                change = max(-self.slew[i] * dt, min(self.slew[i] * dt, target[i] - self.output[i]))
                self.output[i] = max(-self.limits[i], min(self.limits[i], self.output[i] + change))
            return self.output[:], ready, reason
