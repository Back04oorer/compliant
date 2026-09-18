# Follower MIT 重力前馈（2026-09-12）

## 启动行为

既有 TUI「Robot 端主线 / 臂 + 3cam + RViz」仍走原启动脚本和
`robot_setup/start_robot_all.launch`，无需新增 slot。
2026-09-12 水瓶接触点实验前，按用户最新要求将默认值恢复为
`enable_follower_gravity:=true`：下次以 MIT 模式启动双臂时自动开启 follower
重力补偿，保留原 Kp/Kd；显式传入 `enable_follower_gravity:=false` 可关闭。
启用时同时启动：

- 左右 follower 专用驱动适配层 `piper_ctrl_follower_node.py`；节点名不变。
- 一个 `follower_gravity_compensation.py` 进程，为左右臂分别计算 G(q)。

已做右臂一个固定姿态的开启/关闭实机对照；尚未覆盖其他姿态或完成模型标定。
本次默认值修改不重启/热切换机械臂。再次显式启用会改变实际前馈力矩，
须清空工作空间、卸下额外负载、准备急停后低风险验证。

`enable_follower_gravity:=false` 恢复原 `piper_ctrl_node.py` 和无重力前馈配置；
`piper_ctrl_mode:=p` 或 `enable_dual_arm:=false` 不启动这套补偿。
这些是启动参数，不提供不受控的热切换。
既有 host 启动脚本的 `--` 后可传递 `enable_follower_gravity:=false`。

## 保持不变的部分

共享 baseline/leader 的 `piper_ctrl_node.py` 未改。Follower 保留 Kp=7、Kd=0.8、
50 Hz 控制 / 100 Hz state 发布、原 `joint_cmd_mux` 位置目标、位置滤波和夹爪逻辑。
没有添加第二个 CAN 命令发送进程，没有把 leader 低刚度/力反馈配置用于 follower。
不订阅示教/策略 action 的 effort；`enable_tor=false`，不增加摩擦补偿。
新的额外力矩只有 `t_ref = guarded(scale * G(q))`，由原 MIT 发送路径叠加。

## 参数和契约

统一配置：`config/follower_gravity.yaml`。亦可通过 `follower_gravity_config:=路径` 指定副本。

| 项目 | 当前值 |
| --- | --- |
| 双臂每关节 scale | `[0.28, 0.28, 0.28, 0.95, 0.95, 0.95]` |
| 基座坐标系重力 | 两臂均 `[0, 0, -9.81]`，假设直立安装 |
| URDF | `piper_description/urdf/piper_description.urdf` |
| 计算目标频率 | 100 Hz（墙上时钟循环，非实时调度保证） |
| state / 消费侧源时间戳超时 | 0.10 s |
| 渐入时间 | 计算端、消费端各 0.5 s，并有独立变化率限制 |
| 前馈幅值限幅 | SDK t_ref 单位 `[3, 3, 3, 2, 2, 2]` |
| 前馈变化率 | SDK t_ref 单位/秒，六关节均 4 |

Scale 沿用已存在的 teleop 计算初值，**不是已标定的 follower Nm 转换**。
确认基座方向、关节符号、质量/质心、工具负载和 SDK/固件 t_ref 单位后才能认为它准确。
限幅可减少异常输出，但不能证明这些力矩对任意姿态、碰撞或负载都安全。

读取 state 时按 `joint1..joint6` 名称映射，夹爪总开度按 `+width/2,-width/2`
填入 URDF 两个手指关节。关节角不按 URDF limit 截断（保持实测几何）；
检查有限值、弧度/米的大致范围。输出为六个命名 effort，position/velocity 为空。

每臂的话题（以左臂为例）：

- 输入：`/robot/arm_left/joint_states_single`。
- 计算值：`/robot/arm_left/follower_gravity_cmd`，stamp 保留源 state 的时间。
- 计算端状态：`/robot/arm_left/follower_gravity_ready`。
- 驱动当前前馈缓冲值：`/robot/arm_left/follower_gravity_applied`，不是电机实测力矩或 CAN ACK。
- 驱动有效输入/渐入状态：`/robot/arm_left/follower_gravity_active`；true 不等于标定通过。

右臂替换为 `arm_right`。旧 `/robot/arm_*/joint_states_compensated`
仍保留给原 leader 管线，不会双重接入。

## 异常处理与边界

源 state 缺失/过期/乱序/NaN、计算器退出、重复旧帧（即使 ROS 时钟冻结）时，
消费侧独立 watchdog 让前馈以变化率限制退回零；原位置控制继续。
不是断电或急停。满幅时，过期后最多还需约 0.75 s（前三关节）/0.5 s（后三关节）
衰减到零；正常 50 Hz 调度还会有一帧级误差。线程卡住时不承诺该时长。
禁止/重新使能及 P→MIT 时重新渐入。程序启动为 P 模式后再改 MIT，不会自动加载本适配层。
ROS 时间倒退时采取拒绝旧顺序的保守行为；修正系统时钟后可能需要重启。

原 state 发布用当前 ROS 时间而不是硬件采样时间。因此驱动另检查公开 SDK
`GetArmJointMsgs().time_stamp` 是否持续推进，覆盖 CAN 整体停更的情况。
该 timestamp 是聚合值，**不能证明六个关节各自都新鲜**。
本补偿 watchdog 也不能处理驱动本身崩溃、CAN 写线程停顿、电机断电等硬件故障；
它不替代电机/固件 watchdog、急停或完整的关节安全控制器。

## 下一步实测

仅在有人看护、无额外负载、姿态安全的情况下做 A/B。每组都只采样一次当前 q0，
持续发送同一个 q0，比较无/有补偿的关节误差 `q(t)-q0` 和前馈缓冲值。
不要每周期把新的 q(t) 又当新目标：那样目标会跟着下沉，位置恢复力很小，
测试的不是「固定目标保持」。缓慢选择少量安全姿态，先确认补偿方向正确且误差减少。
若上抬/抖动/明显饱和，停止试验并恢复旧配置；不要直接加大 scale/Kp。

## 离线验证

在已 source `workspaces/scripts/use_robot.sh` 的 ROS 容器中运行：

```bash
python3 $(rospack find piper_ctrl)/tests/test_follower_gravity.py
```

测试不执行节点 init、不连接 ROS master、不构造 SDK 实例、不启动机械臂。
覆盖渐入、限幅、变化率、无输入、进程退出、重复/乱序/过期/非有限消息、
停用/恢复、CAN 聚合停更、真实 MIT 发送方法 + 假 CAN 的回归对照、
URDF 重力计算、源时间戳保留、root/intervention launch 参数和回退配置。
