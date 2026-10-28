"""Prepare a fixed-finger URDF without changing arm frames, masses or inertias."""
import xml.etree.ElementTree as ET
from .config import REPO, OUTPUT


def prepare_urdf():
    description=REPO/'src/piper_ros/src/piper_description'
    source=description/'urdf/piper_description.urdf'
    tree=ET.parse(source);root=tree.getroot()
    for mesh in root.iter('mesh'):
        value=mesh.get('filename')
        if value.startswith('package://piper_description/'):
            mesh.set('filename',str(description/value.split('package://piper_description/',1)[1]))
    for joint in root.findall('joint'):
        if joint.get('name') in ('joint7','joint8'):
            joint.set('type','fixed')
            for child in list(joint):
                if child.tag in ('limit','mimic','axis','dynamics'):joint.remove(child)
    link=ET.SubElement(root,'link',name='tcp')
    inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='0.000001')
    ET.SubElement(inertial,'inertia',ixx='0.00000001',iyy='0.00000001',izz='0.00000001',ixy='0',ixz='0',iyz='0')
    joint=ET.SubElement(root,'joint',name='tcp_fixed',type='fixed')
    ET.SubElement(joint,'parent',link='link6');ET.SubElement(joint,'child',link='tcp');ET.SubElement(joint,'origin',xyz='0 0 0.16',rpy='0 0 0')
    OUTPUT.mkdir(parents=True,exist_ok=True);path=OUTPUT/'piper_fixed.urdf';tree.write(path)
    return path
