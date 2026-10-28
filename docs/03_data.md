# 3. 数据采集、转换和单臂抽取

TUI 的 session、数据集、JSON 和输出目录均支持列表浏览、`/` 筛选与 Tab 手输路径；
Esc 返回，输入错误留在当前项修改。转换仍拒绝活动录制、BAD_ROSBAG 和已存在的输出。
操作说明见 [TUI 统一选择与输入](01_startup.md#tui-selection-and-input)。

## 采集

```bash
python3 scripts/teleop_tui.py --execute   # 主菜单 r，在实时预览弹窗里选择相机角色
# 同一功能的命令行预览（添加 --execute 才启动）：
python3 scripts/piper.py record --session my_task \
  --camera left=/dev/video0 --camera top=/dev/video2
```

SPACE 开始/停止 episode；K 删除最近一条；Q 结束录制、等待保存完成并退出。
录制在前台显示 ACTIVE、episode 和时长（每 0.5 秒刷新）；停止后在前台显示 SAVING 和
保存校验阶段的进度条，输出 PASS/ALERT 后才回到 READY，随后可开始下一条。
保存未结束时禁止开始下一条；保存期间按下的空格会被丢弃，READY 后需重新按空格。
相机选择弹窗同时显示实时画面，每路下方可选择 Left wrist / Right wrist / Top / Skip。
点击 Confirm（或 Enter）继续，Cancel / Esc / 关闭窗口取消；未选相机不会录制。
相同角色重新分配时会替换旧相机，避免左右标签混淆。预览设备全部释放后才启动正式录制。
Left wrist / Right wrist / Top 可任选一个、两个或三个；每个视角需要一个独立的已映射相机。
当前 `/dev/video0`、`/dev/video2`、`/dev/video4` 三台设备均可用；未映射的第三台
通过宿主机相机访问服务直接交给容器内驱动，无需重建容器或停止机械臂。默认输出 680×480、30 FPS；本机使用原生 640×480 MJPEG，
驱动在发布压缩 RGB 前缩放为 680×480。图像未标定，不生成深度。
每次打开录制相机时自动设置 UVC 自动曝光（`auto_exposure=3`），包括驱动重开设备时。
因此断电重插后无需手动调曝光；仍需确保设备映射有效或宿主机相机访问服务正在运行。
未选视角不录制也不检查。还录制双臂 robot state（position/effort/velocity）、teleop state、
**机器人实际 mux 命令作为 action 标签**。原始 bag 不提交到 Git，不自动上传或删除。
`metadata.json`/`BAD_ROSBAG.txt` 记录健康状态；PASS 是采集质量检查，不代表任务成功。
不把背景数据处理完成当作下一条物体已摆好的证明。

session 根目录保存 `recording_cameras.json`（设备、角色、输出尺寸、原生尺寸、话题）
和 `recording_profile.json`。同名 session 不能混入不同相机映射或尺寸，变更时使用新 session。
上述路径直接启动所选相机，退出录制器并保存完成后关闭；不要另开驱动占用同一设备。
不带 `--camera` 的旧命令仍使用原三路录制 profile，并要求其相机话题已运行。

本机双相机实测：分别映射为 Left wrist 与 Top 时，两路 ROS 压缩图均约 30 FPS，
测试 bag 中 JPEG 均为 680×480，且不包含未选的 Right wrist。录制器也完成了保存和设备释放。
该测试未启动机械臂数据流，完整健康检查会因缺少 arm topics 标记 BAD；
正式训练数据仍须在遥操运行时录制并通过完整检查。

## TUI 中转换普通 RGB 数据

主菜单 `c` 提供普通 RGB 转换表单，调用同一个转换器，输出双臂 state28/action14、
30 Hz、mux 标签和 LeRobot v3 格式。新 session 根据 `recording_cameras.json` 只转换选中的
RGB 视角，默认沿用录制尺寸。旧 session 没有该文件时仍手动填写：`mono` 输入真实单目压缩图话题；
旧双目拼接顶部图选择 `sbs`，提供 calibration JSON 与 processing metadata JSON，
仅进行标定裁剪，不生成深度或加载外部模型。左右腕图使用当前录制 profile 的话题。
缺少/过期标定不能靠直接缩小整幅拼接图代替。

TUI 会拒绝覆盖已有输出，拒绝仍有 `.bag.active` 或带 `BAD_ROSBAG.txt` 的 session。
转换使用 `--training-python` 指定的独立训练环境；预览模式只打印命令。
其他字段选择、单臂抽取或自定义相机话题仍使用下面的命令行入口。

## 时间对齐

转换器使用公共时间窗口、均匀 30 Hz 采样、最近时间戳匹配与 max_sync_lag 过滤。
这不等于通过图像或物理系统测得的零延迟；转换前看 metadata、掉帧/间隔和对应画面。
`--image-use-bag-time` 改变时间基准，只有确认 header 错误后才用，不作为通用修复。
`workspaces/scripts/lfv_audit_image_timing_rosbag.py` 在 ROS Python 环境审计图像时间。
不要把带 BAD 标记的 episode 直接混入训练；放入单独审核后的 session 目录再转换。

## 推荐：旧双目 → 统一 RGB + 米制深度

在独立训练环境中（见下一篇），准备自己的标定、FastFS、DINO/DeFM 资源。
`calibration.json` 与 `processing_metadata.json` 必须与相机、裁剪一致：

```bash
python training/scripts/convert_rosbag_to_lerobot.py \
  --session-dir /data/clean_session --output-root /data/lerobot --dataset-id my_task_metric \
  --task "task description" --fps 30 --action-source mux --convert-to-v30 \
  --left-wrist-topic /fisheye_left/usb_cam_node/image_raw/compressed \
  --right-wrist-topic /fisheye_right/usb_cam_node/image_raw/compressed \
  --obs-indices all --state-indices 1-14,22-35 --action-indices all \
  --image-width 640 --image-height 320 \
  --actfs --actfs-depth-output-format defm_metric \
  --actfs-calibration-json /assets/calibration.json \
  --actfs-processing-metadata /assets/processing_metadata.json \
  --actfs-model-dir /assets/fastfs.pth \
  --actfs-fast-foundationstereo-root /assets/Fast-FoundationStereo \
  --actfs-valid-iters 8 --actfs-scale 0.5161290322580645 \
  --actfs-max-disp 64 --actfs-max-depth-m 2.5

python training/scripts/materialize_depth_mmap.py \
  --source /data/lerobot/my_task_metric --output /data/lerobot/my_task_mmap --dtype float16
```

该路径先按标定去畸变/裁剪并按 FastFS 契约生成深度，RGB/深度保持几何一致。
不要把整幅 side-by-side 原图简单缩成 640×320 当作相同的单目策略输入。
非双目 RGB ACT 可不加 `--actfs`，但必须显式传正确单目 top topic 和适合的宽高。
RealSense 原生 Z16 深度转换不在这个旧双目 FastFS 示例中；不能直接冒充同一管线。

## 字段顺序

原始 42-D 状态：L position7 + effort7 + velocity7，接着 R 的相同三段。
`--state-indices 1-14,22-35` 是 **1-based** 选择，得到 state28（含两臂 effort，不含 velocity）。
Action14 是左右各 joint6 + gripper1。数据转换/训练必须保持顺序和单位一致。

左单臂例子：

```bash
python training/scripts/build_active_arm_lerobot_subset.py \
  --source-dataset-dir /data/lerobot/my_task_mmap \
  --output-root /data/lerobot --dataset-id my_task_left --active-arm left \
  --episode-indices '[0,1,2]'
python training/scripts/validate_dataset.py \
  --dataset-root /data/lerobot/my_task_left --arm-mode single --active-arm left --variant fused
```

输出 state14/action7；保留 Top 与活动臂腕图，删除另一臂字段。
这里 `[0,1,2]` 只是前三条的示例，必须换成想训练的全部 **0-based** episode 索引。
**物理左腕被映射到 canonical `observation.right_wrist_image`**，这个键名不代表加入物理右腕。
双臂则 state28/action14 与双腕；深度 float16 mmap 保存的是米制数值，不是彩色深度可视化。

新入口保留原 `lfv_teleop_top_stereo_raw_profile.yaml`：腕部 usb_cam 使用私有命名空间
`/fisheye_*/usb_cam_node/image_raw`，与原动态 RViz 配置一致；Top 自定义节点是
`/top_stereo/color/image_raw/compressed`。不要仅凭 launch 中非私有 remap 判断腕图实际名称。
更换驱动后先用 rostopic list / bag info 确认实际 topic，并同步修改 profile 与转换参数。
