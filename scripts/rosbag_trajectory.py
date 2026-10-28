#!/usr/bin/env python3
"""Read only the two selected JointState streams from one finalized ROS1 bag."""
import argparse
import json
import math
from pathlib import Path

from rgb_act_execution import finite_vector
from rgb_act_ros_observer import JOINT_NAMES


def read_trajectory(path, source):
    import rosbag

    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix != '.bag':
        raise ValueError('Choose one finalized .bag file')
    if source not in {'mux', 'state'}:
        raise ValueError('Trajectory source must be mux or state')
    suffix = 'joint_cmd_mux' if source == 'mux' else 'joint_states_single'
    topics = {side: '/robot/arm_' + side + '/' + suffix for side in ('left', 'right')}
    samples = {side: {'timestamps': [], 'positions': []} for side in topics}
    sides = {topic: side for side, topic in topics.items()}
    with rosbag.Bag(str(path), 'r') as bag:
        for topic, msg, bag_time in bag.read_messages(topics=list(sides)):
            names = list(msg.name)
            if len(names) != 7 or set(names) != set(JOINT_NAMES):
                raise ValueError('Unknown/duplicate joint names on ' + topic)
            positions = finite_vector(list(msg.position), 7, topic + ' positions')
            positions = [positions[names.index(name)] for name in JOINT_NAMES]
            stamp = msg.header.stamp.to_sec()
            if stamp == 0:
                stamp = bag_time.to_sec()
            if not math.isfinite(stamp) or stamp <= 0:
                raise ValueError('Invalid timestamp on ' + topic)
            stream = samples[sides[topic]]
            if stream['timestamps'] and stamp <= stream['timestamps'][-1]:
                raise ValueError('Non-increasing timestamps on ' + topic)
            stream['timestamps'].append(stamp)
            stream['positions'].append(positions)
    if any(len(stream['timestamps']) < 2 for stream in samples.values()):
        raise ValueError('Bag requires both selected left/right streams: ' + ', '.join(topics.values()))
    return {'schema_version': 1, 'bag': str(path), 'source': source, 'topics': topics,
            'joint_names': JOINT_NAMES, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--source', choices=('mux', 'state'), default='mux')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trajectory = read_trajectory(args.bag, args.source)
    with args.output.open('x') as output:
        json.dump(trajectory, output, allow_nan=False)
    print('Read-only trajectory extraction: ' + str(args.output), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit('Rosbag trajectory rejected: ' + str(error))
