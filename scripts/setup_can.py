#!/usr/bin/env python3
"""Discover and configure four USB-CAN roles. No CAN frames or arm enable commands."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("robot_left_can", "robot_right_can", "teleop_left_can", "teleop_right_can")


def inventory():
    links = json.loads(subprocess.check_output(["ip", "-details", "-json", "link", "show"], text=True))
    for link in links:
        if link.get("link_type") == "can" or link.get("linkinfo", {}).get("info_kind") == "can":
            device = Path("/sys/class/net") / link["ifname"] / "device"
            link["usb_port"] = device.resolve().name if device.exists() else None
    return links


def ensure_drivers_stopped(interfaces, proc_root=Path('/proc')):
    drivers = ('piper_ctrl_', 'piper_teleop_', 'piper_gravity_', 'follower_gravity_')
    launches = {'start_robot_all.launch', 'start_teleop_all.launch',
                'piper_dual_robot.launch', 'piper_dual_teleop.launch'}
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            names = [Path(arg).name for arg in path.read_bytes().decode(errors='replace').split('\0') if arg]
        except FileNotFoundError:
            continue
        if any(name.startswith(drivers) or name in launches for name in names):
            raise ValueError(f'CAN controller is running (PID {path.parent.name}); stop drivers and compensation first.')
    # Host-network receive lists also expose CAN users outside this container.
    devices = set(interfaces) | {'any'}
    for path in (proc_root / 'net/can').glob('rcvlist*'):
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in devices:
                raise ValueError(f'CAN socket is active on {fields[0]}; stop CAN users before initialization.')


def plan(cfg, links):
    if cfg.get("hardware_reviewed") is not True:
        raise ValueError("Set hardware_reviewed=true only after physically verifying all four arm roles.")
    names = [cfg.get(role, "") for role in ROLES]
    ports = [cfg.get("can_usb_ports", {}).get(role, "") for role in ROLES]
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,14}", name) for name in names) or len(set(names)) != 4:
        raise ValueError("Four unique valid CAN interface names are required.")
    if any(not re.fullmatch(r"\d+-[\d.]+:\d+\.\d+", port) for port in ports) or len(set(ports)) != 4:
        raise ValueError("Four unique verified USB interface paths are required (example: 1-2.3:1.0).")
    if type(cfg.get("can_bitrate")) is not int or cfg["can_bitrate"] != 1000000:
        raise ValueError("This Piper profile requires CAN bitrate 1000000.")
    commands, mapping = [], []
    occupied = {link["ifname"] for link in links}
    for role, name, port in zip(ROLES, names, ports):
        found = [link for link in links if link.get("usb_port") == port]
        if len(found) != 1:
            raise ValueError(f"{role}: USB path {port} has {len(found)} CAN matches; all four must be present.")
        link = found[0]
        current = link["ifname"]
        if name != current and name in occupied:
            raise ValueError(f"{role}: target name {name} already belongs to another interface. No automatic swaps.")
        bitrate = link.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate")
        up = "UP" in link.get("flags", [])
        mapping.append({"role": role, "usb_port": port, "current": current, "target": name, "up": up, "bitrate": bitrate})
        if up:
            if current != name or bitrate != cfg["can_bitrate"]:
                raise ValueError(f"{role}: active interface differs from requested configuration; stop drivers and resolve manually.")
            state = link.get("linkinfo", {}).get("info_data", {}).get("state")
            if state != "ERROR-ACTIVE":
                raise ValueError(f"{role}: CAN state {state!r} is not healthy; no automatic reset.")
            continue
        commands.append(["ip", "link", "set", "dev", current, "type", "can", "bitrate", str(cfg["can_bitrate"])])
        if current != name:
            commands.append(["ip", "link", "set", "dev", current, "name", name])
        commands.append(["ip", "link", "set", "dev", name, "up"])
    return commands, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Read-only kernel CAN/USB inventory; no config required")
    parser.add_argument("--config", type=Path, default=ROOT / "config/hardware.local.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--drivers-stopped", action="store_true", help="Operator confirms all CAN drivers and controllers are stopped")
    parser.add_argument("--check-drivers-stopped", action="store_true", help="Check controller processes and CAN sockets before activation (TUI)")
    args = parser.parse_args()
    try:
        links = inventory()
        if args.list:
            print(json.dumps([link for link in links if "usb_port" in link], indent=2))
            return 0
        cfg = json.loads(args.config.read_text())
        commands, mapping = plan(cfg, links)
        print(json.dumps(mapping, indent=2), flush=True)
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        for command in commands:
            print(shlex.join(sudo + command), flush=True)
        if not args.execute:
            print("PREVIEW ONLY. No interfaces changed; no robot commands sent.")
            return 0
        if not (args.drivers_stopped or args.check_drivers_stopped):
            raise ValueError("Stop robot/leader/gravity controllers before activation; then pass --drivers-stopped.")
        # Recheck before the first mutation; refuse if interface state changed.
        if plan(cfg, inventory()) != (commands, mapping):
            raise ValueError("CAN inventory changed after preflight; review and retry.")
        if commands and args.check_drivers_stopped:
            ensure_drivers_stopped([item['current'] for item in mapping])
        for command in commands:
            subprocess.run(sudo + command, check=True)
        remaining, checked = plan(cfg, inventory())
        if remaining:
            raise ValueError("Post-activation check failed: some interfaces are still down.")
        print(json.dumps(checked, indent=2))
        print("PASS: four CAN interfaces up at 1 Mbit/s. This does NOT verify motor communication or enable arms.")
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"CAN setup stopped: {exc}\nNo automatic rollback/reset. If activation began, inspect interfaces before retrying.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
