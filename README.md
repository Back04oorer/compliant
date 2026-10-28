# compliant

双臂 Piper ROS1 Noetic 的启动、遥操作采集、重力补偿、LeRobot 数据转换与 ACT 训练代码整理。
提供独立 catkin 工作空间、单容器运行方式和默认只预览的操作入口。

| 流程 | 入口 | 说明 |
|---|---|---|
| 构建 | `scripts/bootstrap_ros.sh` | 为本仓库建立独立 catkin workspace |
| roscore / follower / leader | `scripts/piper.py` | 默认仅打印命令；必须显式 `--execute` |
| 终端菜单 | `scripts/teleop_tui.py` | 遥操、rosbag 录制、RGB 数据转换、ACT 与 Piper SoftMimic 仿真训练 |
| follower 重力补偿 | `piper_ctrl/launch/piper_dual_robot.launch` | MIT driver + 有失效保护的补偿前馈 |
| leader 重力补偿 | `scripts/piper.py teleop` | 随示教臂驱动一起启动，不写 follower 位置 |
| 遥操采集 | TUI 主菜单 `r` | 自选相机组合、RGB、state/effort、mux action、健康报告 |
| ROS bag → LeRobot | `training/scripts/convert_rosbag_to_lerobot.py` | 时间戳对齐、可选视角与尺寸、可选 FastFS depth |
| 单臂抽取 | `training/scripts/build_active_arm_lerobot_subset.py` | 去除非活动臂状态/动作/腕图 |
| 标准 RGB ACT | `scripts/train_act.py` | ResNet ACT，不等于 Stereo-ACT 或 DINO-only |
| Stereo-ACT | `training/scripts/train_policy.py --action-head act` | DINO + DeFM；默认 batch 16 / 80k |
| 单臂柔顺仿真 | [SoftMimic / PPO](sim/piper_softmimic/README.md) | IsaacLab 并行训练；夹爪固定；受推、让开、撤力恢复 |

## 从这里开始

0. [环境安装配方与四路 CAN 配置](docs/00_installation.md)（含 ROS Dockerfile、SDK/训练安装脚本）
1. [环境、CAN、启动与遥操](docs/01_startup.md)
2. [两种重力补偿与安全边界](docs/02_gravity.md)
3. [采集、时间对齐、数据转换](docs/03_data.md)
4. [ACT / Stereo-ACT 训练](docs/04_training.md)
5. [来源、许可证与已验证范围](docs/05_provenance.md)
6. [Top 双目鱼眼相机标定](docs/06_camera_calibration.md)（TUI 主菜单 `k`）

```bash
git clone git@github.com:pairlab-usyd/piper-robot-config-and-act-training.git
cd piper-robot-config-and-act-training
python3 scripts/piper.py robot     # 只查看命令，不接管机器人
python3 scripts/check_repo.py     # 离线静态检查，不启动 ROS/训练
```

## 日常启动：进入 Docker → TUI → 遥操

以下命令使用容器名 `piper-control`；如果使用其他名称，请统一替换。宿主机需要先按
[四路 CAN 配置](docs/00_installation.md)
确认四路硬件角色和 USB 地址，日常通过 TUI 的 `i` 初始化接口。
首次使用或代码更新后，先看下一节的镜像构建与容器创建。

ROS master、follower/leader、相机和录制均在同一个 `piper-control` 容器中运行，
通过多个终端分别操作，不需要另一个容器提供 master。
容器启动不会自动启动 ROS master 或机械臂；进入 TUI 后先选 `1` 启动 master，
已有可用 master 时跳过，不重复启动。

本机已完成单容器迁移，旧容器已删除，**只保留 `piper-control`**。
ROS master、TUI、机械臂驱动和相机录制入口都在该容器中，无需启动其他配套容器。

当前 ROS master 已在 `piper-control` 后台运行，日志是容器内 `/tmp/piper-roscore.log`，
可以直接跳过菜单 `1`。这是本次部署的运行状态，**重启容器后不会自动恢复 master**。
本机已保存带最新录制与自动曝光功能的镜像 `piper-ros-noetic:camera-recording`；
`docker ps` 显示的镜像名仍可能是容器创建时的基础镜像名。

### 1. 在宿主机进入容器

先查看容器状态：

```bash
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

已有容器为 `Exited` 时，先在宿主机启动：

```bash
docker start piper-control
```

然后进入容器。容器已经运行，或者需要打开第二、第三个终端时，也使用同一条命令：

```bash
docker exec -it -w /opt/piper piper-control bash
```

每个终端都进入同一个容器。用 `exit` 关闭这种 `docker exec` shell 不会停止整个容器。
容器未配置开机自启；宿主机重启后，需要先 `docker start piper-control`，再进入 TUI 启动 master。

下面的命令均在容器内运行。每个终端先进入仓库，载入 ROS 工作空间：

```bash
cd /opt/piper
source workspaces/scripts/use_robot.sh
```

注意当前终端所在环境：Docker 内仓库路径是 `/opt/piper`，镜像已构建 ROS1 Noetic 工作空间。
宿主机仓库和容器内仓库是两份独立源码，宿主机的 ROS2 Jazzy 不能代替容器中的 ROS1 Noetic。
如果在宿主机打开 TUI 并启动 follower，会检查宿主机自己的工作空间并报未构建；
先执行上面的 `docker exec`，在容器内重新打开 TUI，不需要为了这条提示去宿主机编译 ROS1。
只有容器内 `/opt/piper/workspaces/ws_robot/devel/setup.bash` 也缺失时，才在容器内执行
`bash scripts/bootstrap_ros.sh` 和 `bash workspaces/scripts/build_robot_ws.sh`，再重新载入。
不要在运行机械臂时重新构建或更新依赖。

### 2. 打开菜单并分别启动驱动

先用预览模式查看菜单和将执行的命令：

```bash
python3 scripts/teleop_tui.py
```

退出预览后，实际操作使用：

```bash
python3 scripts/teleop_tui.py --execute
```

菜单使用分区边框、模式标识和刷新动画；绿色表示 UP / 已使能，黄色表示未知 / 部分使能，
红色表示缺失 / 故障，未使能显示为暗色。状态始终保留文字，不支持颜色的终端仍可使用。
可按上下方向键移动高亮、回车选择，也可以直接按数字或字母快捷键，无需回车。
推荐终端至少 80 列 × 24 行；小窗口会滚动菜单，放大窗口可看全硬件状态。
已有可用 ROS master 时，用两个终端分别运行 follower 和 leader。每个终端都进入
同一个容器，在 `/opt/piper` 目录执行对应命令：

```bash
# 终端 B：follower，包含执行臂重力补偿
python3 scripts/piper.py robot --execute
# 终端 C：leader，包含示教臂重力补偿
python3 scripts/piper.py teleop --execute
```

两组驱动都在前台运行，日志直接显示在各自终端；**在对应终端按 Ctrl-C 停止该组驱动和补偿**。
也可以分别打开 TUI：终端 B 选主菜单 `2` → 子菜单 `1`，终端 C 选主菜单 `2` → 子菜单 `2`。
TUI 同样以前台方式运行驱动，Ctrl-C 停止后按回车返回菜单。
文件和目录可列表浏览、`/` 筛选或 Tab 手输路径，固定选项用方向键选择；Esc 返回，输错留在当前项修改。
相机保留实时预览；完整按键与取消规则见 [TUI 统一选择与输入](docs/01_startup.md#tui-selection-and-input)。
两组驱动启动时自动使能；需要监测、重新使能或录制时，可另开终端 D 使用 TUI，保持 B、C 中的驱动运行。
服务不存在时，Enable 会提示应先在另一个终端启动哪组驱动。
如果还没有 ROS master，在终端 A 通过主菜单 `i` 初始化 CAN，再选 `1` 启动 master；
已有可用 master 时跳过 `1`。

示教臂补偿使用 `config/hardware.local.json` 的 `gravity_joint_scale`。
**无需另外启动 `gravity-teleop`**，不要重复运行补偿节点或另一套 CAN 驱动。

`i. Initialize CAN (4 ports)` 读取 `config/hardware.local.json` 中的 `can_usb_ports`，
按 USB 地址匹配设备，自动重命名为配置的 `can_sl / can_sr / can_ml / can_mr` 并以 1 Mbit/s 启动。
PREVIEW 只显示计划，`--execute` 模式下选择 `i` 才实际配置，不需要额外输入初始化命令。
它不依赖 ROS master，初始化会检查本容器内的驱动/补偿进程和 CAN 接收订阅，检测到占用时拒绝修改。
四路必须全部匹配；重复地址、名称冲突、BUS-OFF 或已启动但配置不符时不会自动重置。
已经正确启动的接口保持原状；此选项不启动驱动、不使能机械臂。
USB 重插或宿主机重启后，在驱动停止时重新选择 `i` 即可。

顶部状态在主菜单和子菜单中都会自动刷新：

| 显示 | 含义 |
|---|---|
| `CAN: UP (ERROR-ACTIVE)` | 接口已启动，CAN 控制器处于正常工作状态；不代表已收到电机反馈或已使能 |
| `CAN: DOWN` / `MISSING` | 接口未启动 / 配置的接口不存在 |
| `CAN: UP (BUS-OFF)` | 接口已启动，但总线故障，需要排查 |
| `DRIVER: RUNNING` / `STALE` / `NO DATA` | 收到新鲜驱动诊断 / 诊断超过 2 秒 / 未收到有效诊断；无数据时不能确认驱动是否运行 |
| `CTRL: ENABLED` / `DISABLED` / `UNKNOWN` | 驱动内部控制开关；与电机实际使能反馈分开显示，诊断过期或缺少字段时为未知 |
| `ARM: ENABLED (6/6)` / `DISABLED (0/6)` / `PARTIAL (n/6)` | 这条臂六个关节电机的使能反馈，不包含夹爪 |
| `GRIP: ENABLED` / `DISABLED` | 夹爪独立的使能反馈 |
| `UNKNOWN` | 无法确认状态：ROS/驱动不可用、反馈缺失、过期，或旧驱动未发布新增字段 |

### 3. 自动使能，确认跟随状态

follower、leader 启动后自动使能，无需逐次选择菜单 `3`～`6`。**启动即允许电机出力并执行
控制目标，机械臂可能运动，夹爪可能闭合。** 启动前确认周围无人、主从姿态接近。
启动后查看四个 `CTRL: ENABLED`、`ARM: ENABLED (6/6)`；下表使能操作保留用于重试，
菜单 `7` 可重新选择 teleop 命令来源。

| 子菜单按键 | 操作 |
|---|---|
| `3` | 使能左 leader |
| `4` | 使能右 leader |
| `5` | 使能左 follower |
| `6` | 使能右 follower |
| `7` | 将左右 follower 的命令来源切到 teleop，接通遥操跟随 |

每次命令结束后按回车返回菜单，检查相应 `ARM` / `GRIP` 反馈。
使能服务返回成功只检查六个关节，夹爪是否已使能仍需看 `GRIP`。
`b` 返回主菜单；`q` 退出当前菜单，不会停止其他终端中的驱动，也不是急停。驱动在其运行终端按 Ctrl-C 停止。

### 4. 可选：采集数据

主菜单 **`p. Preview all cameras (live)`** 直接弹出 OpenCV 实时预览窗口，
自动平铺当前映射到容器的所有相机，
显示设备编号和接收帧率；缺失、忙碌或画面过期时显示提示。无需选择录制角色，也不会开始录制。
先确保 ROS master 已启动；窗口与录制相机选择器共用 Docker 的桌面显示配置。
按窗口里的 **Q / Esc / Close**，或点击窗口关闭按钮即可返回 TUI；Ctrl-C 也会等待释放预览自己打开的设备。
空闲相机由预览临时打开，开始录制前先关闭该预览窗口。

`m` 自动启动或复用推理相机，首次角色绑定使用独立相机命令，见[预测指南](docs/07_rgb_act_predictions.md#live-observations-still-predictions-only)。
预测和执行均使用 `2` → `1` 的 follower 驱动提供状态；启动驱动会自动使能机械臂。
录制已经运行时，可在另一个容器终端选择 `p`，预览会订阅已有相机图像，关闭预览不会停止录制。
无需停止 follower 或 leader；已打开的旧 TUI 需在新的终端重新启动才能看到新增菜单。

在终端 D 按 `b` 返回主菜单，再按 `r`，输入 session 名称，例如 `my_task`。
随后自动弹出相机选择窗口，**同时显示各台相机的实时画面**。
直接在每路画面下点击 **Left wrist / Right wrist / Top / Skip**，按实际安装位置指定角色。
至少选择一个视角，未选择的相机不录制；同一角色只能分配给一台相机，重新选择会替换旧分配。
点击 **Confirm** 或按 Enter 确认，点击 **Cancel**、关闭窗口或按 Esc 取消。
预览关闭并释放设备后才进入录制器，忙碌或没有新鲜画面的设备不能用于确认。
宿主机已识别三台相机：`/dev/video0`、`/dev/video2`（DECXIN）和 `/dev/video4`（USB2.0 Camera RGB）。
当前容器保留前两台的设备映射；第三台通过宿主机相机访问服务直接交给容器内驱动，
**无需重建容器或停止机械臂**。预览和录制选择器会合并两种来源，显示全部可用相机。
本机已启用 `piper-camera-access.service` 用户服务，登录后自动启动；
使用方法见 [运行中新增相机](docs/00_installation.md#运行中新增相机无需重建容器)。
已打开的旧 TUI 需在新终端重开，运行驱动的终端保持原样。
程序不能从设备编号判断安装位置，请按实物对应 Left/Right/Top。

默认输出 **680×480、30 FPS、压缩 RGB**，以 640×480 MJPEG 采集后缩放到 680×480，
不做去畸变或标定。
相机在进入录制器时启动，退出且保存完成后关闭；无需设置 `enable_cameras=true`，
也无需另开相机驱动。每次打开或重新打开相机都会设置自动曝光（`auto_exposure=3`），
断电重插后重新进入录制即可应用，前提是对应设备已映射，或宿主机相机访问服务正在运行。
旧的三相机 launch 不应同时占用这些设备。
未选视角不录制，也不参与相机健康检查；双臂状态、teleop 状态和 mux action 仍须正常发布。

左右腕录制示例：session 填 `my_task`，在弹窗中按实物画面分别指定 Left wrist、Right wrist，
点击 Confirm，尺寸直接回车使用 `680x480`。修改已有 session 的相机映射时需使用新 session 名称。
弹窗需要本地桌面显示，Docker 显示配置见 [安装指南](docs/00_installation.md)。
仅启动相机不能开始正式遥操采集：如果没有 follower/teleop 状态和 mux action 话题，
按空格会预检失败，不会生成正式 episode；先完成前面的驱动启动并确认所需话题正常。

进入录制器后：

- **空格**：开始 / 停止一条 episode。
- **K**：删除当前 session 最近一条 episode；先停止录制再删除。
- **Q**：结束录制，等待当前保存完成后退出。

录制器在当前终端运行，每 0.5 秒显示 `ACTIVE`、episode 名称和已录时长。
按空格停止后进入 `SAVING`，前台持续显示文件收尾、统计、健康检查和 metadata 保存阶段的进度条，
最后显示 `PASS` 或 `ALERT`。完成后才显示 `READY`，可以开始下一条；录制和保存都不脱离当前终端。
保存期间的空格不会排队触发下一条，看到 `READY` 后需要重新按空格。

数据写入 `/opt/piper/rosbag_data/<session名>/`。按下一节的方式挂载后，文件会保存在
宿主机仓库的 `rosbag_data/`。健康报告 `PASS` 表示采集检查通过，不代表任务成功；
数据转换与训练见 [数据指南](docs/03_data.md) 和 [训练指南](docs/04_training.md)。

已验证左右两路相机约 8 秒的单独测试录制，每路 230 帧、680×480、30 FPS，
rosbag 完整保存且图像均可解码。测试时没有机械臂数据，所以诊断包标记为 BAD，
保存在 `rosbag_data/camera_diagnostics/`，不能作为训练数据；完整遥操采集尚待驱动运行后验收。

### 5. 在 TUI 中转换数据与训练普通 RGB ACT

普通 RGB ACT 的只读 ROS 观测与动作预测入口见 [Predictions-only adapter](docs/07_rgb_act_predictions.md)；该入口不会发送机器人控制指令。
要绕过 policy、用机械臂检查录制轨迹，使用 TUI **`b. Replay rosbag on robot (local ARM)`**，
在另一终端通过 `2` → `1` 启动共享 follower 驱动；已有驱动直接复用。
本地 ARM 自动将 mux 切到 policy，停止/退出恢复 teleop，无需重启驱动。操作与预检规则见 [rosbag 实机回放](docs/10_rosbag_replay.md)。
独立的 [Supervised ACT execution adapter](docs/08_rgb_act_execution.md) 默认 dry run，实际执行还需双重执行标志和本地终端 ARM；采用共享 follower/mux 和 J2/J3 的录制 MIT 规则，其余运动与反馈检查保留。
在 `piper-control` 容器内运行 `python3 scripts/teleop_tui.py --execute` 后选择 **`m. Test trained RGB ACT`**，
checkpoint 直接回车即可使用最近保存的完整模型，再选择 `predictions` 和 `cuda`，运行 10 次预测并返回菜单，不发送机器人动作；也可选择 `cpu`。
相机在 `m` 进入时自动启动，已有图像流直接复用；预测和执行均需已启动的从臂驱动提供状态。
三路相机绑定保存在 `outputs/tui/inference_cameras.json`；首次没有配置时通过[独立相机命令](docs/07_rgb_act_predictions.md#live-observations-still-predictions-only)配置一次，之后无需单独启动相机。
需要 `robot` / `dry-run` 或手动选择 CPU 时，使用 **`m. Test trained RGB ACT`**。模型、观测和执行桥全部在容器内运行，
推理环境为 `/opt/piper-inference`，安装与迁移见 [Docker 推理指南](docs/09_docker_inference.md)。`m` 的 robot/dry-run 模式先读取数据集 episode 0、frame 0 的实测关节与夹爪位置；
使用 `2` → `1` 的共享从臂驱动（选项 9 已移除）；三路相机由推理入口自动启动或复用，本地输入 ARM 后缓慢定位，到位稳定后持续使用最新观测执行 ACT。
按 TicTac 逻辑，每轮 `select_action()` 输出一步，使用 temporal ensemble（0.01），不再等 50 点执行完。
J2/J3 按录制 MIT 的 ±12.5 rad 协议范围检查，保留负 J2 / 正 J3；其他关节、夹爪、速度及反馈检查保留。
定位阶段最多 60 秒；空格暂停/继续，Enter 缓慢返回数据集首帧并保持，空格重新推理。
`q` 或首次 Ctrl-C 会先缓慢返回数据集首帧，再退出；第二次 Ctrl-C 中止返回。用 `q` 发起返回时，按两次 Ctrl-C 可中止。
返回期间保留速度、加速度、反馈和定位超时检查；故障时立即停止，不自动回位。ARM 前退出不移动机械臂。
暂停和复位会清空策略缓存并丢弃进行中的旧预测，并保持 policy 控制。ARM 自动切换 mux，停止/退出恢复 teleop；
恢复前先对齐 leader，恢复后从臂会跟随 leader。手动 `2` → `7` 可接管回遥操；推理入口不自动启动驱动或改零位。

主菜单新增三个入口，均沿用当前 PREVIEW / EXECUTE 模式：

| 按键 | 功能 | 需要填写 |
|---|---|---|
| `c` | rosbag → 普通 RGB LeRobot v3 数据集 | session 目录、新数据集目录、任务描述、顶部图格式与话题、输出尺寸 |
| `s` | 普通 RGB ACT 小规模验证 | 数据集目录、新输出目录；执行 batch=1、step=1 的真实训练检查 |
| `t` | 普通 RGB ACT 正式训练 | 数据集目录、新输出目录、训练步数、batch size；默认 80,000 步、batch 16 |

这些入口使用独立的 Python 3.10–3.12 训练环境，默认路径为仓库内的
`.venv-training/bin/python`。环境安装见 [安装指南](docs/00_installation.md)；也可在已配好
训练环境和 GPU 的主机/容器中启动同一个 TUI，显式指定解释器路径：

```bash
python3 scripts/teleop_tui.py --execute --training-python /path/to/training-env/bin/python
```

将示例路径替换为实际环境中的 Python 文件。它必须在运行 TUI 的同一主机/容器中可访问，
不是宿主机另一环境的路径；数据集、输出目录、标定文件也必须在该环境可访问。
**基础 ROS 镜像只提供采集依赖；推理扩展镜像提供 RGB ACT 推理，训练仍需独立训练环境。** TUI 会在执行前检查
Python 版本和所需模块，缺依赖时返回菜单；不会自动安装依赖，也不会用 ROS Python 包覆盖训练环境。
在训练主机没有 ROS/CAN 时，硬件状态显示 UNKNOWN，不妨碍使用数据与训练入口。

`c` 生成双臂 state28/action14、30 Hz、mux action 标签的数据集。
新录制的 session 会保存 `recording_cameras.json`，转换器表单据此选择已录制的 RGB 视角，
默认沿用 680×480；普通 ACT 也使用数据集中实际存在的相机。
没有该文件的旧 session 仍需显式确认顶部图格式：

- `mono`：已是单目 RGB，填写实际的压缩图像话题。
- `sbs`：当前顶部双目拼接原图；填写匹配设备的 calibration JSON 和 processing metadata JSON，
  使用现有标定裁剪成单目 RGB。此入口不生成深度、不加载 FastFS/DINO/DeFM 模型。

转换前需结束录制并等待保存；发现 `.bag.active` 或 `BAD_ROSBAG.txt` 时拒绝转换，先单独整理经过复核的
可用 session。已有数据集或训练输出目录均拒绝覆盖；失败留下的输出也应先检查，再换新目录重试。
训练根据数据集维度自动选择 single 或 bimanual；`s` 与 `t` 使用不同输出目录。
已有的单臂 state14/action7 数据可直接选择；`c` 本身不会将双臂数据自动抽成单臂。
完成表单后会打印完整命令；PREVIEW 只展示，EXECUTE 才运行。转换和训练占用当前终端，
退出后按回车返回菜单。TUI 不包含 Stereo-ACT 入口。

### 6. Piper SoftMimic 仿真训练（主菜单 `f`）

在已安装 Isaac Sim 的 GPU 宿主机中重开 TUI：

```bash
python3 scripts/teleop_tui.py --execute
```

选择 **`f. Piper SoftMimic training (simulation)`**，再选择：

- `status`：查看最近的训练状态、更新次数和最新检查点；默认选项。
- `resume`：选择 SoftMimic PPO 检查点和新输出目录，恢复策略、优化器与学习率。
- `new`：选择 CMA 数据版本和新输出目录，从随机权重训练；默认选择最近完成的数据集。
  `dataset_balanced_v1` 已平衡保持/受推/恢复并覆盖小力 XYZ 工况，奖励和 PPO 参数保持原值。
- `evaluate`：独立评估两个检查点与解析基线。默认选最近训练的最新保存权重，
  参考权重优先选同一 run 的 `model_5500.pt`；确认冻结的 XYZ 协议、噪声模式和新输出目录。
  默认同时评估有/无观测噪声，结束后输出正负 XYZ 对比图、全部工况 CSV 和 JSON。

该入口使用独立的 `.venv-softmimic/bin/python`，可用 `--softmimic-python` 指定
同一宿主机/容器内的 Python 3.11 Isaac 环境；基础 ROS 容器不包含该环境。
默认不带 `--execute` 时只预览命令。训练采用已验证的批量力场优化、固定夹爪、
4096 并行环境与 100,000 次总更新预算，详细方法见
[SoftMimic 适配说明](sim/piper_softmimic/README.md)。

训练在当前终端显示日志。按 Ctrl-C 后，等待当前 PPO 更新完成，保存策略与优化器检查点再返回菜单。
续训会校验 URDF 模型、增强数据、观测维度和 PPO 参数，并使用新目录保留旧运行。
检查点不含物理仿真状态和随机数状态；续训会重新初始化仿真。
续训默认使用原运行记录的数据路径；旧版 `FirstPPo` 继续使用原 `dataset`。
修改数据分布需要创建新 run，不能用 `resume` 将旧优化器悄悄切换到新数据。
评估不会恢复优化器或停止正在运行的训练；Ctrl-C 只取消当前评估。
评估结果保存在 `outputs/piper_compliance/softmimic_official/evaluations/`，
网页可通过打印的 `/media/evaluations/<目录>/xyz_positive.png` 路径查看。
首次缺少固定评估协议时，按 [独立评估说明](sim/piper_softmimic/README.md#independent-pushrecovery-evaluation) 生成一次。

### 6. 结束操作

先按录制器的 `Q` 等待保存完成。支撑好机械臂、准备硬件急停后，在 follower、leader
各自的运行终端按 `Ctrl-C` 停止驱动和补偿，再停止自己启动的 ROS master。
停止软件不保证电机已经失能。
**停止补偿、失能或退出驱动可能导致机械臂下垂，不等于保持姿态。**

确认录制已保存、驱动和补偿均已退出后，在宿主机停止唯一运行容器：

```bash
docker stop piper-control
```

停止容器会停止其中的 ROS master 等全部进程；`docker stop` 不应代替正常的录制保存和驱动退出。
下次启动仍按 `docker start` → `docker exec` → TUI 菜单 `i` → `1` 的顺序操作。

## 首次部署或更新 Docker 镜像

下面的命令在宿主机的仓库根目录运行。新机器先安装 Docker，并按
[安装指南](docs/00_installation.md) 准备宿主机 CAN 工具。

```bash
cp -n config/hardware.example.json config/hardware.local.json
python3 scripts/setup_can.py --list
```

按实物填写 `hardware.local.json` 中四路 CAN 的角色、唯一 USB 地址和需要的相机参数，
复核后设置 `hardware_reviewed=true`。默认角色是左/右 follower：`can_sl` / `can_sr`，
左/右 leader：`can_ml` / `can_mr`。USB 地址必须实测，不能照抄示例。

```bash
python3 scripts/setup_can.py                  # 先预览，确认四路匹配
# 可选的宿主机初始化；日常也可在容器 TUI 中选择 i
# 仅在已停止所有控制进程、机械臂已支撑好时执行
python3 scripts/setup_can.py --execute --drivers-stopped

docker build -f docker/Dockerfile.ros -t piper-ros-noetic:local .
docker build -f docker/Dockerfile.inference -t piper-ros-noetic:inference .
```

镜像包含构建时的源码和已编译工作空间。**宿主机改代码或重建镜像，不会更新已有容器中的源码。**
更新后应从新镜像创建新容器；若旧容器仍叫 `piper-control`，可给新容器命名为 `piper-control-new`，
并同步替换本文命令里的容器名。先备份旧数据，不要直接删除旧容器。

首次创建容器时，先确认配置文件存在，再挂载配置和采集数据目录：

```bash
test -f config/hardware.local.json
mkdir -p rosbag_data data outputs
docker run -it --network host --name piper-control \
  --gpus all \
  --cap-add=NET_ADMIN \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video2:/dev/video2 \
  --device=/dev/video4:/dev/video4 \
  --mount "type=bind,src=$PWD/config,dst=/opt/piper/config,readonly" \
  --mount "type=bind,src=$PWD/rosbag_data,dst=/opt/piper/rosbag_data" \
  --mount "type=bind,src=$PWD/data,dst=/opt/piper/data" \
  --mount "type=bind,src=$PWD/outputs,dst=/opt/piper/outputs" \
  piper-ros-noetic:inference
```

这是同机 Linux SocketCAN 的控制容器示例：CAN 通过 host 网络可见，进入容器不会自动启动机械臂。
`--cap-add=NET_ADMIN` 允许容器内的 TUI 初始化 CAN；无需 `--privileged` 或宿主机 sudo 密码。
旧容器缺少该权限时，仅重启无法补上，需要按上述参数重新创建并保留原数据挂载。
推理镜像默认包含 CUDA 12.8，宿主机需安装 NVIDIA 驱动和 NVIDIA Container Toolkit。
此命令映射 GPU 和当前三台 USB 相机，未映射 X11。设备编号须按本机实际情况调整；
无相机部署可移除对应 `--device`。Docker 设备映射在创建时生效，
旧容器仅 `docker restart` 不会增加设备。运行中新增相机可使用上述宿主机访问服务，
USB 重插或编号变化后重新打开预览或选择器核对画面；无需为此中断机械臂。
默认关闭 RViz。
镜像已在 `/opt/piper` 内编译工作空间，不需要再将整个宿主机仓库覆盖挂载到该路径。

| 容器路径 | 宿主机仓库目录 | 持久化方式 |
|---|---|---|
| `/opt/piper/config` | `config/` | 只读挂载；在宿主机修改硬件配置 |
| `/opt/piper/rosbag_data` | `rosbag_data/` | 读写挂载，保留录制数据 |
| `/opt/piper/data` | `data/` | 读写挂载，保留转换后的数据集 |
| `/opt/piper/outputs` | `outputs/` | 读写挂载，保留训练输出 |

源码不在上述挂载中，容器内 `/opt/piper/scripts` 等目录不会随宿主机编辑自动同步。
重建容器时继续挂载同一批目录；数据不会因停止容器消失，但不要在备份前删除未挂载的数据。

旧容器如果没有挂载 `rosbag_data`，录制文件只保存在容器内部。删除或替换旧容器前，
在确认已有录制目录的情况下，从宿主机备份（容器停止时也可以复制）：

```bash
backup_dir="$(mktemp -d "$HOME/piper-rosbag-backup.XXXXXX")"
docker cp piper-control:/opt/piper/rosbag_data/. "$backup_dir/"
```

训练源码快照来自 `pairlab-usyd/Stereo-patch` commit
`26290bcd247f1fb373a517a7057f66fd40ef1198`；ROS 源码来自本机当前配置。
训练目录还保留其已有 DP / Flow V2 实现，本仓库的主流程以 ACT 为主。

不包含数据集、rosbag、模型权重、token、SSH 密钥、训练日志或个人云端地址。
URDF 网格是机器人描述的必要资产，不是训练数据。
不复制原 TUI 的大量任务专用部署槽位；提供任务无关的新入口。
`src/` 内历史脚本不都是推荐入口，优先按本文档操作。

**安全与许可：**容器启动本身不会启动机械臂；`robot`、`teleop` 驱动入口会自动使能，
不自动回零，默认不启动相机或旧手眼 TF；
TUI 的录制入口才会启动选中的相机。
不是碰撞安全控制器；stop/失能不等于保持姿态，断力矩可能下垂。
部分原 ROS 包许可证仍为 TODO，所以先作为私有仓库整理，未赋予整库统一开源许可证。
