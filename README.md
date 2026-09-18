# Piper 机械臂配置和 ACT 训练

双臂 Piper ROS1 Noetic 的启动、遥操作采集、重力补偿、LeRobot 数据转换与 ACT 训练代码整理。
从当前工作代码复制到独立仓库，**不会修改原机器人部署**。

| 流程 | 入口 | 说明 |
|---|---|---|
| 构建 | `scripts/bootstrap_ros.sh` | 为本仓库建立独立 catkin workspace |
| roscore / follower / leader | `scripts/piper.py` | 默认仅打印命令；必须显式 `--execute` |
| follower 重力补偿 | `piper_ctrl/launch/piper_dual_robot.launch` | MIT driver + 有失效保护的补偿前馈 |
| leader 重力补偿 | `scripts/piper.py gravity-teleop` | 与 follower 分开，不写 follower 位置 |
| 遥操采集 | `scripts/piper.py record --session TASK` | RGB、state/effort、mux action、健康报告 |
| ROS bag → LeRobot | `training/scripts/convert_rosbag_to_lerobot.py` | 时间戳对齐、640×320、可选 FastFS depth |
| 单臂抽取 | `training/scripts/build_active_arm_lerobot_subset.py` | 去除非活动臂状态/动作/腕图 |
| 标准 RGB ACT | `scripts/train_act.py` | ResNet ACT，不等于 Stereo-ACT 或 DINO-only |
| Stereo-ACT | `training/scripts/train_policy.py --action-head act` | DINO + DeFM；默认 batch 16 / 80k |

## 从这里开始

0. [环境安装配方与四路 CAN 配置](docs/00_installation.md)（含 ROS Dockerfile、SDK/训练安装脚本）
1. [环境、CAN、启动与遥操](docs/01_startup.md)
2. [两种重力补偿与安全边界](docs/02_gravity.md)
3. [采集、时间对齐、数据转换](docs/03_data.md)
4. [ACT / Stereo-ACT 训练](docs/04_training.md)
5. [来源、许可证与已验证范围](docs/05_provenance.md)

```bash
git clone git@github.com:pairlab-usyd/piper-robot-config-and-act-training.git
cd piper-robot-config-and-act-training
python3 scripts/piper.py robot     # 只查看命令，不接管机器人
python3 scripts/check_repo.py     # 离线静态检查，不启动 ROS/训练
```

训练源码快照来自 `pairlab-usyd/Stereo-patch` commit
`26290bcd247f1fb373a517a7057f66fd40ef1198`；ROS 源码来自本机当前配置。
训练目录还保留其已有 DP / Flow V2 实现，本仓库的主流程以 ACT 为主。

不包含数据集、rosbag、模型权重、token、SSH 密钥、训练日志或个人云端地址。
URDF 网格是机器人描述的必要资产，不是训练数据。
不复制原 TUI 的大量任务专用部署槽位；提供任务无关的新入口。
`src/` 内历史脚本不都是推荐入口，优先按本文档操作。

**安全与许可：**默认不自动使能、不自动回零、不启用相机或旧手眼 TF。
不是碰撞安全控制器；stop/失能不等于保持姿态，断力矩可能下垂。
部分原 ROS 包许可证仍为 TODO，所以先作为私有仓库整理，未赋予整库统一开源许可证。
