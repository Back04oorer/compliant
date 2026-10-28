# Top 双目鱼眼标定

TUI 的标定板预设用方向键选择，标定目录可列表浏览、筛选或手输；参数输错留在当前项修改。
相机继续实时预览选择 Top；取消预览或在表单按 Esc 均可返回菜单。
通用按键说明见 [TUI 统一选择与输入](01_startup.md#tui-selection-and-input)。

TUI 主菜单 `k. Calibrate Top stereo fisheye` 启动
[Back04oorer/fisheye_calibration](https://github.com/Back04oorer/fisheye_calibration)
的 ChArUco 网页界面，用于**同一台 Top 相机输出左右拼接画面**的内参及双目外参标定。
两台独立腕相机不能使用这个入口当作一套固定双目。

## 安装

在运行 TUI 的同一环境中，准备 Git、uv 和可访问的相机：

```bash
bash scripts/setup_calibration.sh
```

安装脚本把上游固定到 `6ea4206ab2b1279171ee04a27990178b9992db57`，保存在
`external/fisheye_calibration`，并用其 `uv.lock` 创建独立 Python 3.12 环境。
安装时应用仓库内 `patches/fisheye-bottom-ab.patch` 和
`patches/fisheye-charuco-7x5-38.patch`；重复安装识别已应用的补丁，旧安装会补上新板预设。
已存在但版本不同或有补丁以外修改的源码会停止安装，不会重置。
uv 不在 PATH 时可通过 `UV_BIN=/absolute/path/to/uv` 指定。
Python 3.12 不存在时，uv 可能下载对应解释器。

标定环境与 ROS/训练环境分开；不向 ROS Python 安装 NumPy/OpenCV。
`external/` 已被 Git 忽略，上游源码和虚拟环境不随本仓库提交。
宿主机、容器是独立环境，需要分别安装；已有容器也要同步新的 TUI、适配器及相机读取器。

## 使用

```bash
python3 scripts/teleop_tui.py --execute
```

先启动 ROS master（用于与录制相同的 OpenCV 相机选择窗口），再选 `k`。
窗口同时显示可用相机的实时画面，在实际 Top 相机画面下点击 **Top**，再点 **Confirm**
或按 Enter。只能选择一台 Top；选择另一台会替换之前的选择。无新鲜画面或设备忙时不能确认。
Esc、Cancel 或关闭窗口会取消本次操作。
确认后预览会释放相机，再填写原生左右拼接总分辨率、帧率、标定板预设、输出目录和端口。
默认 `2560x720` 表示每眼 `1280x720`，只是上游默认值，必须选择设备实际支持的模式。
设置错误会显示实际分辨率不匹配，不会静默缩放。

默认网页为 `http://127.0.0.1:8765`。宿主机启动会尝试打开浏览器；容器内启动会显示地址，
在宿主机浏览器打开。本机 `piper-control` 使用 host 网络；其他容器网络配置需要相应连通性。
服务只监听本机回环地址。TUI 默认 PREVIEW 模式仍打开实时选择窗口（与录制入口一致），
选完相机后仅显示标定命令，不启动标定网页或保存标定数据。
终端运行期间保持打开；Ctrl-C 停止服务并释放它打开的相机。

页面使用上游流程：

1. `New session` 新建，或 `Open` 恢复已有会话；`Replay` 只读查看。
2. 核对相机左右顺序、分辨率、标定板布局及实测格长。
3. 默认使用下方提供的新板 PDF；也可用 `Print this board` 导出所选预设。
   按 100% / Actual size 打印，关闭适应页面和缩放，测量后固定在平整硬板上。
4. `Apply settings` 后启动 `Start guided capture`，按照画面提示移动和倾斜标定板，
   保持双目相机固定，提示静止时稳定持板。
5. 程序分别保存拟合图和独立验证图。通过检查后导出 NPZ、YAML 和报告。

Bottom 已拆为 **Bottom A（40%, 80%）** 和 **Bottom B（60%, 80%）**，
坐标指左眼单幅图像中的标定板中心，从左上角计量。采集和独立验证都要分别覆盖两个点，
原 Lower-left、Lower-right 和 Tilt bottom edge 保留。
A/B 的归一化接受半径为 0.08，两个区域不重叠；其他位置仍为 0.14。
页面蓝圈与判定范围一致，左右两眼均需能识别足够角点。
重启标定服务后生效，仅刷新浏览器不能更新服务端代码。
已有采集记录不会删除；Open 恢复会根据原图观测重新计算覆盖，
旧 Bottom 记录不会自动算作 A、B 两点通过。Replay 保留旧报告。
旧会话若已进入验证阶段，不会自动补做全部拟合点；
要确保拟合与验证都完整使用新流程，请新建会话。

默认标定板已更新为 [stereo-charuco-7x5-38-a4-v1（A4 横向 PDF）](assets/calibration/stereo-charuco-7x5-38-a4-v1_A4.pdf)。
TUI 和新建网页会话默认使用此预设；行列是方格数，不是内角点数：

| 预设 | 方格数 | 格长 / marker 边长 | 字典 | legacy |
|---|---|---|---|---|
| `stereo-charuco-7x5-38-a4-v1`（默认） | 7 × 5 | 38 / 30 mm | `DICT_APRILTAG_25h9` | false |
| `existing-20mm` | 11 × 8 | 20 / 15 mm | `DICT_5X5_100` | true |
| `a4-30mm` | 8 × 6 | 30 / 22.5 mm | `DICT_5X5_100` | true |

新板图案尺寸为 266 × 190 mm，标签 ID 为 0–16，共 24 个内角点。
打印后核对 PDF 中横、竖两条 100 mm 标尺；网页 Setup 应显示 7 × 5、38 / 30 mm、
`DICT_APRILTAG_25h9` 和 modern 布局。不要混用旧板的字典、格长或采集图像。
更新后重新运行安装脚本并重启标定服务；已有会话继续读取原有板参数，
使用新板应通过 `New session` 新建会话。

参数必须与实际打印板一致；使用其他板时在网页 Setup 修改并核对。
独立标定解释器可通过 TUI 的 `--calibration-python /absolute/path/to/python` 指定。

## 相机与保存位置

适配器复用现有 V4L2 MJPEG 读取器和宿主机相机描述符服务，支持直接映射设备与
当前容器的宿主机共享设备。标定使用原生拼接尺寸，不调用普通录制的
`640x480 → 680x480` 缩放，也不主动修改曝光。相机在新会话/恢复时才打开。
正在被预览或录制占用的设备会报错；先正常关闭占用它的窗口或录制。
无需停止机械臂驱动来释放未被驱动使用的相机。

默认结果在仓库 `data/calibration/top_stereo/`。容器中对应
`/opt/piper/data/calibration/top_stereo/`，本机已有 data 挂载会保存到宿主机。
每次会话包含 `session.json`、`frames/` 和 `results/<candidate>/`，沿用上游保存及续做机制。

输出 `R,T` 把左相机坐标转换到右相机坐标，双目平移使用毫米。
内参和畸变只对应本次相机模式；当前普通 RGB 录制仍使用原有分辨率与处理方式，
不能未经像素映射验证直接套用高分辨率标定结果。
本入口保存标定资产，不自动改写录制器的 camera_info 或 ACT 数据转换配置。

上游页面还包含其原机器人的固定相机手眼功能，其机器人路径和模型尚未适配本 Piper 仓库；
本入口验收范围是 Top 双目内参/外参和独立图像验证。
上游几何检查通过不等于已经验证毫米级深度或抓取精度。
