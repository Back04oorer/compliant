# 环境安装与四路 CAN

仓库提供可上传、可复用的安装配方，不打包本机系统目录、驱动二进制或个人密钥。
安装脚本默认只预览；添加 `--execute` 才修改环境。不会自动运行 roscore、使能机械臂或开始训练。

## 1. ROS Noetic + Piper SDK

推荐新机器用 Docker 隔离 Ubuntu 20.04 / ROS Noetic，不把旧 ROS 依赖装进新系统。
先在宿主机安装 Docker Engine（按 Docker 官方对应系统文档）。在仓库根目录构建：

```bash
docker build -f docker/Dockerfile.ros -t piper-ros-noetic:local .
```

配方使用官方 `ros:noetic-ros-base-focal`，安装 ROS 附加包、动力学库、CAN 工具、
`piper_sdk==0.6.1`、`python-can==4.5.0`，并编译本仓库的 10 个 ROS 包。
构建时不接入 CAN、相机、GPU，也不启动 ROS 节点。镜像尚未推送镜像仓库；这是源码构建配方。
基础镜像已固定 digest，apt/pip 间接依赖尚未全部锁定；Noetic 已停止维护，需要可访问的上游源。

已有 Ubuntu 20.04 / Noetic 环境也可以使用：

```bash
python3 scripts/setup_environment.py ros             # 查看 apt/pip 安装计划
python3 scripts/setup_environment.py ros --execute   # 安装依赖，可能调用 sudo
bash scripts/bootstrap_ros.sh
bash workspaces/scripts/build_robot_ws.sh
```

这个原生入口要求 Noetic 已安装且 ROS apt 源可用；全新机器的 ROS 本体由上述官方镜像提供。
不要把 ROS 安装命令放到训练 Conda/venv 中执行，也不要在正在控制机器人的容器中升级依赖。

不接硬件的镜像自检（导入依赖和实际 follower/补偿模块、离线计算 URDF 重力，不实例化 SDK 控制器）：

```bash
docker run --rm --network none piper-ros-noetic:local bash -lc \
  'source /opt/piper/workspaces/ws_robot/devel/setup.bash; python3 scripts/check_robot_env.py'
```

CAN 激活在宿主机执行。之后可用 host 网络进入控制容器；**此操作不自动启动机器人**：

```bash
docker run --rm -it --network host --name piper-control \
  -v "$PWD/config/hardware.local.json:/opt/piper/config/hardware.local.json:ro" \
  piper-ros-noetic:local
```

先确认挂载的 local.json 文件存在。容器内仓库在 `/opt/piper`；其他终端用
`docker exec -it piper-control bash` 进入。按 [启动指南](01_startup.md) 分别启动。
CAN Socket 通过 host 网络共享；此例不映射 USB 相机、X11、GPU，不开启 RViz。
相机与 GUI 需要按实际设备另外配置权限和挂载，不建议直接用 `--privileged`。

## 2. 四路 CAN：探测 → 人工确认角色 → 预览 → 激活

宿主机先安装 CAN 工具（Ubuntu/Debian）：

```bash
sudo apt-get update
sudo apt-get install -y can-utils iproute2 ethtool kmod
sudo modprobe gs_usb
python3 scripts/setup_can.py --list
cp -n config/hardware.example.json config/hardware.local.json
```

列表来自 kernel/sysfs，不发送 CAN 帧。逐个确认 USB 适配器接的是哪条臂，填写同一份 local.json：

| 字段 | 角色 | 默认接口名 |
|---|---|---|
| `robot_left_can` | 左 follower | `can_sl` |
| `robot_right_can` | 右 follower | `can_sr` |
| `teleop_left_can` | 左 leader | `can_left` |
| `teleop_right_can` | 右 leader | `can_right` |

将四个实际 USB 路径分别填写到 `can_usb_ports` 的同名字段，例如 `1-2.3:1.0`。
示例不是你的真实地址；程序只能匹配接口，**无法自动知道哪根线接左臂/右臂**。
确认后设 `hardware_reviewed=true`，保持 `can_bitrate=1000000`。

```bash
python3 scripts/setup_can.py                          # 仅预览
# 托住/支撑好机械臂，先停止所有 follower、leader、补偿和 policy 控制进程，再执行：
python3 scripts/setup_can.py --execute --drivers-stopped
```

四路全部预检通过后才开始操作。缺适配器、重复地址、接口名冲突、活跃接口配置不符、
BUS-OFF 都拒绝自动更改。配置正确且已 UP 的接口不重置。
激活后再次检查名字、bitrate、UP 和 CAN 错误状态；没有发使能、运动或归零指令。
若中途失败，可能部分接口已 UP；不自动回滚/关闭，查看报错和 `--list` 后人工处理。
`--drivers-stopped` 是操作员确认，不是自动检测驱动进程的保证。
每次重启后可显式重复此流程，不安装自动激活/机器人自启动服务。

## 3. 独立训练环境

在训练主机准备 Python 3.10–3.12、venv、编译工具、FFmpeg 共享库和 NVIDIA 驱动。
例如 Ubuntu 22.04（Python 3.10）：

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv python3.10-dev build-essential ffmpeg git
python3 scripts/setup_environment.py training --python python3.10
python3 scripts/setup_environment.py training --python python3.10 --execute
source .venv-training/bin/activate
python scripts/check_training_env.py --cuda
```

默认建立全新的 `.venv-training`，不修改现有环境；已存在则拒绝覆盖。
安装 Torch 2.10.0 / Torchvision 0.25.0（CUDA 12.8 wheel）、TorchCodec 0.10.0、
LeRobot 0.4.4 和训练依赖，执行 `pip check` 与基础导入检查。
Ubuntu 20.04 系统 Python 3.8 不能用来训练；在另外的训练主机/容器或自行准备的 Python 3.10 环境运行。
TorchCodec 依赖 FFmpeg 共享库，只有静态 ffmpeg 命令行程序并不一定足够。
不自动安装/升级 NVIDIA 驱动；若驱动与 wheel 不匹配，先查 PyTorch 官方兼容信息。
`--torch-index` 可显式改用该 Torch 版本实际支持的官方 wheel 源。

本次发现旧训练依赖 `numpy<2` 与 LeRobot 引入的 rerun-sdk、OpenCV 冲突，
已按本机训练环境修正为 NumPy 2.2.6。**ROS 仍使用独立的系统 NumPy，不能混装。**
核心版本固定，间接依赖尚未全量锁定；依赖解析通过不等于真实训练 smoke 通过。
DINO/DeFM/FastFS 外部源码和许可受限权重仍按
[外部资源指南](../training/docs/external_dependencies.md) 准备。
安装成功后按 [训练指南](04_training.md) 用自己的数据跑真实 `--smoke`，再开长训练。

## 验证边界

本次新增安装计划和 CAN 预检有离线测试；测试 CAN 接口是模拟数据，不会对实物发指令。
训练依赖已进行 Python 3.10/Linux 的解析验证，未安装新的 GPU 训练环境或执行新模型训练。
镜像构建/硬件实际运行的验收结果见 [验证记录](05_provenance.md)，不要把安装命令当作硬件校准完成。
