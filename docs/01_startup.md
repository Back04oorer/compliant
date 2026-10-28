# 1. 环境、启动和遥操

新机器优先看 [环境安装与四路 CAN](00_installation.md)：已提供 ROS 镜像构建配方、
依赖安装脚本及统一四路 CAN 预检/激活入口。下文保留手动安装与单路激活说明。

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
  ros-noetic-controller-manager ros-noetic-control-msgs ros-noetic-hardware-interface ros-noetic-controller-interface \
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

四个角色：follower 左 `can_sl`、右 `can_sr`；leader 左 `can_ml`、右 `can_mr`。
USB 路径必须实测。出现 `No such device` 时不要重试控制命令，先排查 USB-CAN 连接。
本仓库不自动拔插、重置 CAN，也不重启现有 ROS 容器。

复制 `config/hardware.example.json` 为 `config/hardware.local.json`，确认 CAN 与相机型号、
USB 地址后将 `hardware_reviewed` 改成 true。本地文件不会提交到 Git。

## 分终端启动

每个终端进入仓库根目录；命令默认预览，下面显式执行：

也可以使用终端菜单。默认模式只展示已有命令；加 `--execute` 后才会实际执行，仍会经过
`piper.py` 的硬件配置复核。`robot`、`teleop` 启动时会自动使能机械臂及夹爪，
启动前确认周围无人、主从姿态接近。ROS master、驱动（含补偿）和录制均在前台运行，
各占用自己的终端；Ctrl-C 停止该终端启动的程序，不会结束另一终端的驱动。

```bash
python3 scripts/teleop_tui.py
python3 scripts/teleop_tui.py --execute
```

### TUI selection and input

文件、目录、固定选项和参数表单使用共用交互，无需安装额外 UI 依赖。

| 操作 | 按键与行为 |
|---|---|
| 选择列表项 | ↑/↓ 移动，Enter 确认；长列表可用 PageUp/PageDown |
| 按名称筛选 | `/` 开始输入筛选词，Backspace 删除，Ctrl-U 清空；Enter 选择匹配项 |
| 手输文件或目录 | Tab 或 `Enter a path...`；支持粘贴、空格路径、`~`、环境变量和带引号的路径 |
| 浏览目录 | Enter 进入目录，`../` 到父目录；目录选择用 `Select this directory` 确认当前目录 |
| 新建输出位置 | 浏览到父目录后选 `New directory name here...`，或 Tab 输入完整目标路径 |
| 编辑参数 | 默认值高亮，直接输入可替换；←/→、Home/End、Delete/Backspace 编辑，Ctrl-U 清空 |
| 返回或取消 | Esc 返回上一个浏览页；在顶层选择或参数输入中取消本次操作并返回主菜单 |

无效路径、未完成的 checkpoint、不健康的录制 session、错误的数据集结构、已存在的输出、
非有限数值或越界参数会显示原因，并停留在当前项修改，之前已填的项无需重填。
浏览和选择不会创建输出目录；转换和训练仍禁止覆盖已有目录，标定 session 根目录允许复用。
主菜单保留快捷键，Teleop 子菜单也可用 Esc 返回；在主菜单按 `q` 退出。
管道或无交互终端使用逐行输入，可输入 `:back` 取消当前操作；同样支持原位重试。

回放候选来自 `rosbag_data/*/episode_*/*.bag`，只展示已结束的 `.bag`，默认高亮最近修改的文件。
模型候选来自 `outputs/*/checkpoints/*/pretrained_model`，默认高亮最新完整 checkpoint；
转换选择录制 session，训练选择 `data/lerobot/` 下含 `meta/info.json` 的数据集。
所有路径选择均可切到目录浏览或手输其他位置，不受候选目录限制。

相机继续用实时图像预览分配角色。Cancel、Esc、`q` 或关闭预览窗返回菜单，
并清理该预览启动的相机资源；推理选相机时须有 Left、Right、Top 三路新鲜画面才能确认。
取消表单不会启动正式录制、转换、训练、推理或控制程序。

主菜单包含 `1. ROS master`、`2. Teleop`、`i. Initialize CAN (4 ports)`、
`p. Preview all cameras (live)`、`r. Record rosbag session`，以及
`c. Convert rosbag to RGB dataset`、`s. RGB ACT smoke test`、`t. Train RGB ACT`。
容器内 TUI 的 **`m. Test trained RGB ACT`** 可选择训练完成的 checkpoint，运行 dry-run 或本地 ARM 后的持续 ACT；
相机自动启动或复用，共享 follower 驱动需先启动；仅预测时在 `m` 中选择 `predictions` 模式。
入口和结果说明见 [ACT 执行指南](08_rgb_act_execution.md)。
`k. Calibrate Top stereo fisheye` 打开 Top 左右拼接鱼眼相机的引导标定网页；
安装、标定板和结果说明见[相机标定指南](06_camera_calibration.md)。
数据与训练入口需使用独立训练 Python，具体表单和环境设置见 [README](../README.md)。
相机首次绑定使用独立命令，见[预测指南](07_rgb_act_predictions.md#live-observations-still-predictions-only)；`m` 各模式的实测状态均由 follower 驱动提供。
先选 `i` 按配置匹配 USB 地址、重命名并以 1 Mbit/s 启动四路 CAN，再启动驱动。
此操作不依赖 ROS master，不使能机械臂；容器需有 `NET_ADMIN` 权限，检测到驱动或 CAN 订阅占用时拒绝修改。
选择 `2` 进入 Teleop 子菜单：`1` 启动遥操、ACT 和回放共用的 follower 驱动与 mux，`2` 一起启动 teleop（leader）驱动和示教臂重力补偿，
补偿使用硬件配置中的 `gravity_joint_scale`，无需另开补偿终端；其余选项提供逐臂使能与切换遥操跟随。`b` 返回主菜单，`q` 退出 TUI。
选项 9 已移除。ACT／回放在本地 ARM 后自动切到 policy，停止/退出恢复 teleop；暂停和复位保持 policy。
切模式无需重启 follower。`7` 手动切回遥操并使执行桥停止写入，`8` 手动选择 policy（不会启动或 ARM 执行桥）。
恢复遥操前先对齐 leader；恢复后 follower 会接收 leader 的当前位置。
已有可用 master 时，用两个终端分别启动 follower 和 leader；日志直接显示在各自终端。
需要停止时，在对应终端按 Ctrl-C。TUI 停止驱动后按回车返回菜单，不使用后台启动或 `x` 停止入口。
使能和监测可另开一个 TUI，Enable 会先检查对应服务；未启动时提示在其他终端运行
子菜单 `1`（follower）或 `2`（leader）。`q` 只退出当前菜单，不影响其他终端中的驱动。
主菜单与 Teleop 子菜单顶部均自动刷新四路 CAN 和四个 arm 的状态，按菜单键即可选择，无需回车。
CAN 按硬件配置中的接口名显示 `UP (ERROR-ACTIVE)`、`DOWN`、`MISSING` 等；
`UP (BUS-OFF)` 表示接口已启动但总线故障，不能当作通信正常。
`DRIVER: RUNNING/STALE/NO DATA` 表示驱动诊断新鲜、过期或无数据；
`CTRL: ENABLED/DISABLED/UNKNOWN` 显示驱动内部控制开关，与下面的电机反馈分别判断。
这些字段复用现有诊断，只需重新打开 TUI，不需重启机械臂驱动。
ARM 读取现有驱动的 `agent_driver_state` 六个关节电机反馈，显示 `ENABLED (6/6)`、
`DISABLED (0/6)` 或 `PARTIAL (n/6)`；每条臂下另列 `GRIP: ENABLED/DISABLED/UNKNOWN`，
读取夹爪独立的使能反馈，不计入关节的 6/6。ROS/驱动不可用、反馈缺失或超过 2 秒均显示
`UNKNOWN`，不会把无反馈当成未使能。监测只读，不发送 CAN 控制或调用使能服务。
更新代码后需在安全停机后重新启动对应驱动，才会发布新增的夹爪诊断字段；旧驱动显示 `GRIP: UNKNOWN`。
通常每约 1–2 秒更新；查询超时可延长到约 5 秒。终端较窄时可加宽以看全状态。
命令结束后按回车返回菜单；管道输入时每次显示菜单只读取一次状态。

```bash
# A：只运行一个 roscore；已有 master 就不重复启动
python3 scripts/piper.py core --execute
# B：双 follower，MIT + follower gravity；自动使能，不自动回零
python3 scripts/piper.py robot --execute
# C：双 leader + leader 重力补偿，关闭拨片；自动使能
python3 scripts/piper.py teleop --execute
```

启动后检查四个 `CTRL: ENABLED` 和 `ARM: ENABLED (6/6)`，正常时无需手动使能。
以下命令仅用于重新使能或重新选择 teleop 命令来源：

```bash
python3 scripts/piper.py enable-teleop-left --execute
python3 scripts/piper.py enable-teleop-right --execute
python3 scripts/piper.py enable-robot-left --execute
python3 scripts/piper.py enable-robot-right --execute
python3 scripts/piper.py mux-teleop --execute
```

`gravity-teleop` 命令仅保留用于单独调试，不要与已包含补偿的 `teleop` 同时启动。
不要同时运行原 TUI 的另一组驱动。一个 CAN 设备只能有一个控制进程。
不要把 `stop_srv` 当暂停保持。退出/断开补偿前托住机械臂，准备硬件急停。

## 相机

主菜单 `p` 直接弹出 OpenCV 窗口查看全部已映射相机，显示设备编号和帧率，
不选择角色或开始录制。需有可用 ROS master 和 Docker 桌面显示配置。
Q / Esc / Close 或关闭窗口会释放预览自己打开的相机。
已在录制的相机直接订阅其 ROS 图像，不重新打开设备或停止原发布者；空闲相机临时打开，
开始录制前需先关闭占用设备的预览窗口。画面停止更新后会明确显示过期提示。

普通 USB RGB 录制使用 TUI 主菜单 `r`：选择 `l`（Left wrist）、`r`（Right wrist）、
`t`（Top）的任意非空组合，再为每个视角指定不同设备。宿主机当前三台相机对应 `/dev/video0`、
`/dev/video2`、`/dev/video4`；当前容器通过已有映射与宿主机相机访问服务读取三台，
无需停止机械臂或重建容器。预览与选择器均显示所有可用设备。
设备编号不代表实际安装位置。
按 [Docker 创建命令](../README.md#首次部署或更新-docker-镜像) 映射设备后即可使用，
不需要打开硬件配置中的 `enable_cameras`。相机随录制器启动/退出，默认输出 680×480、30 FPS，
本机通过 640×480 MJPEG 采集后缩放，不做标定或去畸变。

以下是独立保留的旧三相机启动方式，不与上述录制相机同时运行：
配置 `enable_cameras=true` 和三个 USB port 后，入口使用旧 Top Stereo：
Top side-by-side 2560×720 / 30 FPS，左右 USB 腕相机 640×480。
这是旧双目配置，不适用于 RealSense。内外参必须与实物匹配，不沿用本机标定到别的设备。
新入口负责启动驱动，**不会应用原 host wrapper 的 v4l2 色彩锁定**；采集前仍需在
实际设备上确认曝光/白平衡与训练数据一致。不要声称 AWB 下的历史图像能靠一个 Kelvin 数字完全复现。
`top_usb_driver_impl=mjpg_direct` 要求设备已经配置为所选 MJPG 分辨率/帧率，
不匹配时拒绝打开；先用 v4l2-ctl 检查设备支持的模式并设置，不默默缩放补救。

不要停止仍在提供 ROS master 或机械臂控制的旧容器；更换容器前先检查正在运行的进程。
