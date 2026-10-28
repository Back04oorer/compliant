"""Export recorded evaluation states and the control URDF for browser playback."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import OUTPUT


def robot_description(path):
    root = ET.parse(path).getroot()
    def vector(node, key, default='0 0 0'):
        return list(map(float, node.get(key, default).split())) if node is not None else list(map(float, default.split()))
    joints = []
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        joints.append(dict(name=joint.get('name'), parent=joint.find('parent').get('link'),
                           child=joint.find('child').get('link'), type=joint.get('type'),
                           xyz=vector(origin, 'xyz'), rpy=vector(origin, 'rpy'),
                           axis=vector(joint.find('axis'), 'xyz', '0 0 1')))
    visuals = []
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:continue
            origin = visual.find('origin')
            visuals.append(dict(link=link.get('name'), mesh=Path(mesh.get('filename')).name,
                                xyz=vector(origin, 'xyz'), rpy=vector(origin, 'rpy'),
                                scale=vector(mesh, 'scale', '1 1 1')))
    return dict(joints=joints, visuals=visuals)


def extract(path, controller, amplitude=2):
    report = json.loads((path/'report.json').read_text())
    assert report['status']=='complete' and not (path/'failure.json').exists()
    with np.load(path/'traces.npz') as data:
        cases = {}
        for i, row in enumerate(report['cases']):
            if row['controller']!=controller or row['variant']!='nominal' or row['force_N']!=amplitude or row['stiffness_N_per_m']!=100:continue
            key = ('+' if row['sign']>0 else '-')+row['axis']
            q, tcp, force = (data[name][:, i] for name in ('q','tcp','force'))
            assert q.shape==(len(data['time']),6) and np.isfinite(q).all() and np.isfinite(tcp).all()
            pre = tcp[(data['time']>=2.5)&(data['time']<3)].mean(0)
            cases[key] = dict(q=q.round(7).tolist(), tcp=tcp.round(7).tolist(), force=force.round(5).tolist(),
                              pre=pre.tolist(), metrics=row)
        assert len(cases)==6
        return dict(cases=cases, time=data['time'].tolist(), reference=data['reference_pos'].tolist(),
                    checkpoint=report['checkpoint_name'], checkpoint_sha256=report['checkpoint_sha256'],
                    protocol_sha256=report['protocol_sha256'])


def main():
    root = OUTPUT/'evaluations'
    comparison = root/'BalancedPush_v1_20000_evaluation_vs8000'
    first = root/'BalancedPush_v1_20000_evaluation_vsFirstPPo/clean_reference'
    modes = {}
    for mode in ('clean','noisy'):
        entries = [('BalancedPush · 20000', comparison/(mode+'_latest'), 'policy', '#2563eb'),
                   ('BalancedPush · 8000', comparison/(mode+'_reference'), 'policy', '#c77723')]
        if mode=='clean':entries.append(('FirstPPo · 24628', first, 'policy', '#9858a0'))
        entries.append(('解析基线', comparison/(mode+'_latest'), 'analytic', '#258369'))
        modes[mode] = [dict(label=label, color=color, **extract(path, controller)) for label,path,controller,color in entries]
        assert all(item['time']==modes[mode][0]['time'] and item['protocol_sha256']==modes[mode][0]['protocol_sha256'] for item in modes[mode])
    output = OUTPUT/'replay';output.mkdir(exist_ok=True)
    payload = dict(robot=robot_description(OUTPUT/'piper_fixed.urdf'), modes=modes,
                   note='Recorded Isaac evaluation states; nominal plant, 2 N, K=100 N/m. Mesh animation uses URDF kinematics; no online policy inference.')
    (output/'data.json').write_text(json.dumps(payload, separators=(',',':'),allow_nan=False))
    print(output/'data.json')


if __name__=='__main__':main()
