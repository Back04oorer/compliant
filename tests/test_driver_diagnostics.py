import importlib.util
import threading
from types import SimpleNamespace

import pytest

from test_entrypoints import ROOT


def diagnostics():
    path = ROOT / 'src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/scripts/piper_agent_diagnostics.py'
    spec = importlib.util.spec_from_file_location('piper_agent_diagnostics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DriverDiagnostics('/robot/arm_left/', clock=lambda: 100.0)


def driver(enabled):
    motor = SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=True), foc_status_code=64)
    piper = SimpleNamespace(
        GetArmStatus=lambda: SimpleNamespace(arm_status=SimpleNamespace(
            ctrl_mode=1, teach_status=0, mode_feed=1, err_code=0, arm_status=0)),
        GetArmLowSpdInfoMsgs=lambda: SimpleNamespace(**{f'motor_{i}': motor for i in range(1, 7)}),
        GetArmGripperMsgs=lambda: SimpleNamespace(gripper_state=SimpleNamespace(
            foc_status=SimpleNamespace(driver_enable_status=enabled))))
    return SimpleNamespace(piper=piper, control_data_lock=threading.Lock(),
        current_joint_positions=[0] * 6, current_joint_velocities=[0] * 6,
        joint_positions_cmd=[0] * 6, joint_states=SimpleNamespace(position=[0] * 7),
        ctrl_mode='p', enable_gripper_haptic=False, p_joint_pos_cmd_topic='/cmd',
        gripper_pos_cmd_topic='/gripper', block_ctrl_flag=False, GetEnableFlag=lambda: True)


@pytest.mark.parametrize('enabled', [False, True])
def test_gripper_diagnostics_report_feedback_and_age(enabled):
    monitor = diagnostics()
    arm = driver(enabled)
    assert monitor.collect(arm)['gripper_feedback_age_s'] is None
    monitor.receive(SimpleNamespace(arbitration_id=0x2A8, data=bytes(8)))
    state = monitor.collect(arm)
    assert state['gripper_enabled'] is enabled
    assert state['gripper_feedback_age_s'] == 0
    assert state['motor_enabled'] == [True] * 6
    monitor.stamps[0x2A8] = 97.0
    assert monitor.collect(arm)['gripper_feedback_age_s'] == 3.0


@pytest.mark.parametrize('fields', [{'is_rx': False}, {'is_error_frame': True},
    {'is_remote_frame': True}, {'is_extended_id': True}, {'data': bytes(7)}])
def test_invalid_gripper_frames_do_not_refresh_feedback(fields):
    monitor = diagnostics()
    message = {'arbitration_id': 0x2A8, 'data': bytes(8)}
    message.update(fields)
    monitor.receive(SimpleNamespace(**message))
    assert monitor.collect(driver(False))['gripper_feedback_age_s'] is None


@pytest.mark.parametrize('arm_status', [0, 7])
def test_arm_status_is_reported_even_without_a_joint_error_code(arm_status):
    arm = driver(True)
    status = arm.piper.GetArmStatus()
    status.arm_status.arm_status = arm_status
    arm.piper.GetArmStatus = lambda: status
    state = diagnostics().collect(arm)
    assert state['arm_status'] == arm_status
    assert state['err_code'] == 0
