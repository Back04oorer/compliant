# 2. 重力补偿

| | Leader 示教臂 | Follower 执行臂 |
|---|---|---|
| 计算节点 | `piper_gravity_compensation_node.py` | `follower_gravity_compensation.py` |
| 驱动 | `piper_ctrl_node.py` | `piper_ctrl_follower_node.py` |
| 输入 | `/teleop/arm_*/joint_states_single` | `/robot/arm_*/joint_states_single` |
| 配置 | 启动参数 gravity_joint_scale | `piper_ctrl/config/follower_gravity.yaml` |
| 启动 | 随 `teleop` 驱动一起启动 | 随 follower MIT launch 一起启动 |

当前继承的缩放系数两侧均为 `[0.28,0.28,0.28,0.95,0.95,0.95]`。
这是现有设备经验值，不是通用准确的牛米换算，换工具/负载/安装方向必须重新验证。

Follower 使用独立前馈 topic，**不替代/重写位置目标**，不单独直接发送 CAN。
`follower_gravity_guard.py` 在驱动端防止陈旧补偿一直保留：

- 100 Hz 计算；state timeout 0.10 s；ramp 0.5 s。
- 绝对 `t_ref` 上限 `[3,3,3,2,2,2]`；变化率上限 `[4,4,4,4,4,4]`。
- 计算端与接收端都做保护/渐变，因此实际建立时间可以超过 0.5 s。
- 保持原 MIT Kp/Kd；基于 URDF 质量与惯性计算，默认假定底座正向直立。

不要另启 leader 的补偿节点来给 follower 做补偿，也不要重复启动 follower 计算器。
断补偿/失能时的下垂风险仍存在；重力补偿不是碰撞检测，也不保证末端绝对精度。
详细实现和原测试见 `piper_ctrl/FOLLOWER_GRAVITY.md` 与 `tests/test_follower_gravity.py`。
