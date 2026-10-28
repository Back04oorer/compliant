# 环境安装与四路 CAN

仓库提供可上传、可复用的安装配方，不打包本机系统目录、驱动二进制或个人密钥。
安装脚本默认只预览；添加 `--execute` 才修改环境。不会自动运行 roscore、使能机械臂或开始训练。

## 1. ROS Noetic + Piper SDK

推荐新机器用 Docker 隔离 Ubuntu 20.04 / ROS Noetic，不把旧 ROS 依赖装进新系统。
先在宿主机安装 Docker Engine（按 Docker 官方对应系统文档）。在仓库根目录构建：

```bash
docker build -f docker/Dockerfile.ros -t piper-ros-noetic:local .
docker build -f docker/Dockerfile.inference -t piper-ros-noetic:inference .
```

配方使用官方 `ros:noetic-ros-base-focal`，安装 ROS 附加包、动力学库、CAN 工具、
`piper_sdk==0.6.1`、`python-can==4.5.0`，并编译本仓库的 10 个 ROS 包。
构建时不接入 CAN、相机、GPU，也不启动 ROS 节点。镜像尚未推送镜像仓库；这是源码构建配方。
基础镜像已固定 digest，apt/pip 间接依赖尚未全部锁定；Noetic 已停止维护，需要可访问的上游源。
推理扩展镜像默认安装 Torch 2.10.0 + CUDA 12.8；宿主机需安装 NVIDIA 驱动和
NVIDIA Container Toolkit，创建容器时必须带 `--gpus all`，镜像本身不能授予 GPU 访问权。

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

确认四路 CAN 的角色和 USB 地址后，可在宿主机激活，或在下述容器内通过 TUI `i` 初始化。
容器使用 host 网络和 `NET_ADMIN` 权限；**进入容器不自动启动机器人**：

```bash
test -f config/hardware.local.json
mkdir -p rosbag_data data outputs
PIPER_XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
test -f "$PIPER_XAUTHORITY"
docker run -it --network host --name piper-control \
  --gpus all \
  --cap-add=NET_ADMIN \
  --env DISPLAY="$DISPLAY" \
  --env XAUTHORITY=/tmp/piper-camera-selector.Xauthority \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
  --mount "type=bind,src=$PIPER_XAUTHORITY,dst=/tmp/piper-camera-selector.Xauthority,readonly" \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video2:/dev/video2 \
  --device=/dev/video4:/dev/video4 \
  --mount "type=bind,src=$PWD/config,dst=/opt/piper/config,readonly" \
  --mount "type=bind,src=$PWD/rosbag_data,dst=/opt/piper/rosbag_data" \
  --mount "type=bind,src=$PWD/data,dst=/opt/piper/data" \
  --mount "type=bind,src=$PWD/outputs,dst=/opt/piper/outputs" \
  piper-ros-noetic:inference
```

先确认挂载的 local.json 文件存在。容器内仓库在 `/opt/piper`；其他终端用
`docker exec -it -w /opt/piper piper-control bash` 进入。已停止的容器用
`docker start -ai piper-control` 启动并进入。采集文件持久化到宿主机的 `rosbag_data/`。
完整的 Docker 进入与 TUI 按键流程见 [README](../README.md)，
或按 [启动指南](01_startup.md) 分别启动。
CAN Socket 通过 host 网络共享；此例映射 `/dev/video0`、`/dev/video2` 和 `/dev/video4` 三台 USB 相机，
设备编号按本机实物调整。无相机时可移除对应 `--device`；
旧容器仅重启不能增加设备；运行中可使用下方相机访问服务，USB 重插后应重新核对编号。此例映射本地 X11 和只读授权文件，
用于录制前的相机预览选择；同时映射 GPU 供默认 CUDA 推理，不开启 RViz，不需要 `--privileged`。
无桌面环境时可省略显示映射，直接用 `piper.py record --camera role=/dev/videoN` 指定相机。

当前已运行的 host 网络容器也可在宿主机配置显示，无需重建或停止机械臂：

```bash
docker cp "${XAUTHORITY:-$HOME/.Xauthority}" piper-control:/tmp/piper-camera-selector.Xauthority
docker exec piper-control chmod 600 /tmp/piper-camera-selector.Xauthority
docker exec piper-control python3 -c 'import json,sys; from pathlib import Path; p=Path("/opt/piper/outputs/tui"); p.mkdir(parents=True,exist_ok=True); (p/"gui.json").write_text(json.dumps({"DISPLAY":sys.argv[1],"XAUTHORITY":"/tmp/piper-camera-selector.Xauthority"}))' "$DISPLAY"
```

相机选择器优先使用环境变量，未设置时读取 `outputs/tui/gui.json`；本机已验证可连接桌面。
TUI `p` 独立实时预览使用 OpenCV 弹窗，与选择器共用上述显示配置。
容器内窗口启动时会编译并加载一个小型 X11 兼容库，对该窗口进程关闭 MIT-SHM，
避免容器与桌面 IPC 隔离导致的显示错误；不需要重启容器或添加 `--ipc=host`。
重新登录桌面导致授权更新，或重建容器后，需要重新配置授权。

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
| `teleop_left_can` | 左 leader | `can_ml` |
| `teleop_right_can` | 右 leader | `can_mr` |

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
日常可直接在容器内运行 `python3 scripts/teleop_tui.py --execute`，选择 `i. Initialize CAN (4 ports)`。
它使用同一配置和预检逻辑，并检查驱动进程与 CAN 接收订阅；正确启动的接口不会重置。
容器必须在创建时添加 `--cap-add=NET_ADMIN`，仅重启已有容器无法补上这项权限。
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

## 运行中新增相机，无需重建容器

宿主机的 `scripts/camera_access.py` 通过已有的 `outputs/` 挂载共享本地 Unix socket，
只允许访问当前枚举到的相机采集节点（不包括 metadata 节点）。
容器内驱动优先打开已映射设备，未映射时请求宿主机打开相机并接收其文件描述符，
随后仍由原有 V4L2 驱动采集、设置曝光和发布 ROS 图像，不经过视频转码中继。
预览和录制选择器都会枚举该服务提供的相机。无需增加 Docker 容器，也无需停止机械臂。

本机已安装并启用 `piper-camera-access.service` 用户服务，随用户登录自动启动：

```bash
# 宿主机执行
systemctl --user status piper-camera-access.service
systemctl --user restart piper-camera-access.service
```

其他机器也可在宿主机仓库内直接运行（保持该终端开启）：

```bash
python3 scripts/camera_access.py --serve
```

宿主机用户须有 `/dev/video*` 的读写权限，容器须挂载同一仓库的 `outputs/`。
服务本身不持续占用相机，退出预览或录制后由相机驱动释放设备。
新增或重插设备后重新打开 TUI 预览/选择器，角色仍按实物画面选择。
旧 TUI 进程需在新终端重开才能加载代码更新，运行机械臂驱动的终端不必关闭。

## 4. 容器内 RGB ACT 推理

推理入口已迁入 `piper-control`，不再使用宿主机 `.venv-training`。
安装独立推理环境或构建 `docker/Dockerfile.inference`，参见 [Docker 推理指南](09_docker_inference.md)。
ROS 保留系统 Python 3.8；模型使用 `/opt/piper-inference/bin/python`。
