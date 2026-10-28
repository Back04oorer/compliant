"""Load unchanged reward functions and forcefield methods from the pinned release."""
import ast
import importlib.util
import subprocess
from .config import UPSTREAM, UPSTREAM_COMMIT


def assert_version():
    actual=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM_COMMIT:raise ValueError('SoftMimic upstream commit changed')
    if subprocess.run(['git','-C',str(UPSTREAM),'diff','--quiet','HEAD']).returncode:
        raise ValueError('Pinned SoftMimic source has local modifications')


def module(name,relative):
    spec=importlib.util.spec_from_file_location(name,UPSTREAM/relative)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def force_methods(namespace):
    path=UPSTREAM/'softmimic_gym/softmimic_gym/tasks/locomotion/tracking/mdp/commands/compliance_augmented_reference_command.py'
    tree=ast.parse(path.read_text())
    wanted={'quat_to_angle_axis','_compute_reactive_forcefield_force_w','_compute_reactive_forcefield_torque_w'}
    nodes=[]
    for node in tree.body:
        candidates=node.body if isinstance(node,ast.ClassDef) else [node]
        for candidate in candidates:
            if isinstance(candidate,ast.FunctionDef) and candidate.name in wanted:
                candidate.decorator_list=[];nodes.append(candidate)
    if {node.name for node in nodes}!=wanted:raise ValueError('Official forcefield methods missing')
    code=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(code)
    exec(compile(code,str(path),'exec'),namespace)
    return namespace
