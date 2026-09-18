# 3. 数据采集、转换和单臂抽取

## 采集

```bash
python3 scripts/piper.py record --session my_task          # 预览
python3 scripts/piper.py record --session my_task --execute
```

SPACE 开始/停止 episode；K 删除最近一条；Q 退出并等后台保存完成。
默认录三路压缩 RGB、双臂 robot state（position/effort/velocity）、teleop state、
**机器人实际 mux 命令作为 action 标签**。原始 bag 不提交到 Git，不自动上传或删除。
`metadata.json`/`BAD_ROSBAG.txt` 记录健康状态；PASS 是采集质量检查，不代表任务成功。
不把背景数据处理完成当作下一条物体已摆好的证明。

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
