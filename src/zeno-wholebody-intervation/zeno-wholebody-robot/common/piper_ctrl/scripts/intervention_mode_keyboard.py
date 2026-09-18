#!/usr/bin/env python3
"""Keyboard publisher for intervention mode commands."""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional

import rospy
from std_msgs.msg import String


class InterventionModeKeyboard:
    def __init__(self) -> None:
        self.mode_cmd_topic = rospy.get_param("~mode_cmd_topic", "/intervention/mode_cmd")
        self.mode_topic = rospy.get_param("~mode_topic", "/intervention/mode")
        self.active_source_topic = rospy.get_param("~active_source_topic", "/intervention/active_source")

        self.mode: Optional[str] = None
        self.active_source: Optional[str] = None

        self.pub = rospy.Publisher(self.mode_cmd_topic, String, queue_size=10)
        rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=10)
        rospy.Subscriber(self.active_source_topic, String, self._active_source_cb, queue_size=10)

    def _mode_cb(self, msg: String) -> None:
        self.mode = str(msg.data).strip()

    def _active_source_cb(self, msg: String) -> None:
        self.active_source = str(msg.data).strip()

    def _status_line(self) -> str:
        mode = self.mode if self.mode else "<unknown>"
        source = self.active_source if self.active_source else "<unknown>"
        return f"mode={mode} active_source={source}"

    def _print_help(self) -> None:
        print("[arbiter_keyboard] hotkeys:")
        print("  i: toggle POLICY <-> HUMAN")
        print("  p: force POLICY")
        print("  h: force HUMAN")
        print("  s: print mode/source")
        print("  q: quit keyboard")

    def run(self) -> None:
        self._print_help()
        print("[arbiter_keyboard] " + self._status_line())

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not rospy.is_shutdown():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue

                key = sys.stdin.read(1)
                if not key:
                    continue

                if key in ("i", "I"):
                    self.pub.publish(String(data="toggle"))
                    rospy.loginfo("[arbiter_keyboard] publish mode_cmd=toggle")
                elif key in ("p", "P"):
                    self.pub.publish(String(data="policy"))
                    rospy.loginfo("[arbiter_keyboard] publish mode_cmd=policy")
                elif key in ("h", "H"):
                    self.pub.publish(String(data="human"))
                    rospy.loginfo("[arbiter_keyboard] publish mode_cmd=human")
                elif key in ("s", "S"):
                    print("[arbiter_keyboard] " + self._status_line())
                elif key in ("q", "Q"):
                    print("[arbiter_keyboard] quit.")
                    return
                elif key == "\x03":
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)


def main() -> None:
    rospy.init_node("intervention_mode_keyboard")
    node = InterventionModeKeyboard()
    node.run()


if __name__ == "__main__":
    main()
