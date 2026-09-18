# 1. 环境、启动和遥操

## 环境分开

- Robot：Ubuntu 20.04 / ROS Noetic / Python 3.8，使用系统 ROS Python。
- Training：单独 Python 3.10+ 环境，CUDA PyTorch、LeRobot 0.4.4；不要覆盖 ROS 的 numpy。
- 本机确认：`piper_sdk==0.6.1`、`python-can==4.5.0`。
- 重力计算需要 **Pinocchio 机器人动力学库**（ROS 包 `ros-noetic-pinocchio`），不是同名的其他 PyPI 包。

在已准备好的 Noetic 主机/容器内安装所需包（需要管理员权限）：

```bash
sudo apt-get update
sudo apt-get install -y can-utils ethtool iproute2 python3-pip python3-numpy python3-yaml \
  python3-rospkg ros-noetic-pinocchio ros-noetic-tf ros-noetic-tf2-ros \
  ros-noetic-topic-tools ros-noetic-robot-state-publisher ros-noetic-joint-state-publisher \
  ros-noetic-xacro ros-noetic-usb-cam ros-noetic-cv-bridge \
  ros-noetic-image-transport ros-noetic-compressed-image-transport ros-noetic-rviz ros-noetic-serial \
  ros-noetic-controller-manager ros-noetic-hardware-interface ros-noetic-controller-interface \
  ros-noetic-realtime-tools ros-noetic-pluginlib
python3 -m pip install piper_sdk==0.6.1 python-can==4.5.0
bash scripts/bootstrap_ros.sh
bash workspaces/scripts/build_robot_ws.sh
```

为满足 bridge 等包的编译依赖，也链接 dm_common/dm_hw/dm_controllers，但不启动拨片。
原代码附带的底盘、达妙拨片运行功能、Piper-X、MoveIt、
RealSense 专用功能不是最小构建的一部分；启用它们需另装依赖并建立对应包链接。
ROS Noetic 已停止维护；新机器的 apt 源/系统镜像需要自行确认，未在空白机器上重新构建镜像。

## CAN 不能按历史 USB 地址盲目启动

```bash
bash find_all_can_port.sh
# 确认每个物理 USB-CAN 与 arm 角色后，逐个激活，例如：
sudo bash can_activate.sh can_sl 1000000 YOUR_USB_PATH
```

四个角色：follower 左 `can_sl`、右 `can_sr`；leader 左 `can_left`、右 `can_right`。
USB 路径必须实测。出现 `No such device` 时不要重试控制命令，先排查 USB-CAN 连接。
本仓库不自动拔插、重置 CAN，也不重启现有 ROS 容器。

复制 `config/hardware.example.json` 为 `config/hardware.local.json`，确认 CAN 与相机型号、
USB 地址后将 `hardware_reviewed` 改成 true。本地文件不会提交到 Git。

## 分终端启动

每个终端进入仓库根目录；命令默认预览，下面显式执行：

```bash
# A：只运行一个 roscore；已有 master 就不重复启动
python3 scripts/piper.py core --execute
# B：双 follower，MIT + follower gravity；不自动使能/回零
python3 scripts/piper.py robot --execute
# C：双 leader，关闭拨片，补偿由下一条单独启动
python3 scripts/piper.py teleop --execute
# D：leader 重力补偿
python3 scripts/piper.py gravity-teleop --execute
```

确认路径无人、两侧姿态不会在跟随切换时产生大跳变，再逐臂使能：

```bash
python3 scripts/piper.py enable-teleop-left --execute
python3 scripts/piper.py enable-teleop-right --execute
python3 scripts/piper.py enable-robot-left --execute
python3 scripts/piper.py enable-robot-right --execute
python3 scripts/piper.py mux-teleop --execute
```

不要同时运行原 TUI 的另一组驱动。一个 CAN 设备只能有一个控制进程。
不要把 `stop_srv` 当暂停保持。退出/断开补偿前托住机械臂，准备硬件急停。

## 相机

默认不启动相机。配置 `enable_cameras=true` 和三个 USB port 后，入口使用旧 Top Stereo：
Top side-by-side 2560×720 / 30 FPS，左右 USB 腕相机 640×480。
这是旧双目配置，不适用于 RealSense。内外参必须与实物匹配，不沿用本机标定到别的设备。
新入口负责启动驱动，**不会应用原 host wrapper 的 v4l2 色彩锁定**；采集前仍需在
实际设备上确认曝光/白平衡与训练数据一致。不要声称 AWB 下的历史图像能靠一个 Kelvin 数字完全复现。
`top_usb_driver_impl=mjpg_direct` 要求设备已经配置为所选 MJPG 分辨率/帧率，
不匹配时拒绝打开；先用 v4l2-ctl 检查设备支持的模式并设置，不默默缩放补救。

在 Docker 内使用时，把这个新仓库单独挂载，网络与 CAN/USB 设备访问由操作者配置。
不要用本仓库命令停止正在工作的旧容器。
