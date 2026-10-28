# 4. ACT 训练

TUI 中用列表选择数据集与输出父目录，也可 `/` 筛选或 Tab 手输路径。
训练步数、batch size 输错后可在当前项修改；Esc 取消返回菜单，已有输出目录仍禁止覆盖。
按键说明见 [TUI 统一选择与输入](01_startup.md#tui-selection-and-input)。

## 环境与资源

推荐使用 [独立训练环境安装入口](00_installation.md#3-独立训练环境)，会固定主要版本、
隔离 ROS Python，并执行 `pip check` 和 TorchCodec 导入检查。下面保留手动安装方式。

新建与 ROS 分开的 Python 3.10+ 环境。按 GPU 驱动安装兼容的 torch/torchvision，再：

```bash
python -m pip install -r training/requirements.txt
python -m pip install -e training
```

依赖版本沿用 Stereo-patch 快照。DINO、DeFM、FastFS 源码/权重单独准备，遵守各自许可证；
参考 `training/docs/external_dependencies.md`。训练不会自行上传 HF，没有嵌入 token。

## Stereo-ACT：先 smoke，再正式训练

```bash
python training/scripts/train_policy.py --action-head act \
  --dataset-root /data/lerobot/my_task_left --output-dir /data/train/my_task_smoke \
  --arm-mode single --active-arm left --variant fused \
  --dino-model /assets/dinov3-vitb16 --defm-root /assets/defm --smoke

python training/scripts/train_policy.py --action-head act \
  --dataset-root /data/lerobot/my_task_left --output-dir /data/train/my_task_act \
  --arm-mode single --active-arm left --variant fused \
  --dino-model /assets/dinov3-vitb16 --defm-root /assets/defm \
  --target-grid 20x40 --chunk-size 50 \
  --batch-size 16 --steps 80000 --checkpoint-steps 60000,80000
```

双臂用 `--arm-mode bimanual` 和双臂数据目录。
`--smoke` 会真实执行 dataloader、forward/backward、optimizer step（batch1/step1），
不是打印参数。`--dry-run` 才只打印命令。smoke 和正式输出目录不能共用。
按代码内 checkpoint hook 保存 60k、80k；检查具体目录及训练日志，不仅看进程是否存在。

DINO-only / DeFM-only：`--variant dino-only` / `defm-only`。
它们仍是 Stereo-ACT 架构的分支消融，不等于下面的标准 ResNet ACT。

## 普通 RGB ACT

训练后的只读 ROS / 离线预测见 [RGB ACT predictions without robot movement](07_rgb_act_predictions.md)。该适配器支持标准 `act` 检查点，默认 CUDA，也可选择 CPU；只保存预测，不控制机器人。
受监督的执行入口见 [Supervised RGB ACT execution](08_rgb_act_execution.md)，使用共享 follower/mux，包含 dry run、本地 ARM、运动与反馈检查。

TUI 主菜单 `s` 运行一次真实的 ACT smoke（batch1/step1），`t` 启动正式训练，
均调用 `scripts/train_act.py`。表单填写数据集和新输出目录；正式训练还可填写步数、batch size。
根据 state14/action7 或 state28/action14 自动识别单臂/双臂，拒绝覆盖已有输出。

```bash
python3 scripts/teleop_tui.py --execute --training-python /path/to/training-env/bin/python
```

需将路径替换为同一主机/容器内已安装依赖的 Python 3.10–3.12 训练环境。
不加 `--execute` 时只打印命令。ROS 容器默认未安装训练环境，也未配置 GPU；
不能直接用 ROS Python 3.8 运行 ACT。TUI 不包含 Stereo-ACT。


```bash
python scripts/train_act.py --dataset /data/lerobot/my_task_left \
  --output /data/train/rgb_act_smoke --arm-mode single --smoke
python scripts/train_act.py --dataset /data/lerobot/my_task_left \
  --output /data/train/rgb_act --arm-mode single --batch-size 16 --steps 80000
```

入口按 meta/info.json 使用实际存在的 Top / 腕部 RGB 视角以及 state，至少需要一路 RGB，
支持 TUI 录制时选择的任意相机组合；不要求补齐未录制的视角，也不把 depth 当成 RGB 图。
旧单臂抽取数据仍沿用 canonical 腕图命名。
普通 RGB 入口设置 `policy.use_amp=true` 和 `ACCELERATE_MIXED_PRECISION=bf16`，
通过 Accelerate 开启 CUDA BF16 混合精度；smoke 与正式训练使用相同精度。
添加 `--image-size 256` 可将所有 RGB 输入缩放为 256×256。缩放在归一化前进行，
设置随检查点的预处理管线保存，部署应加载同一预处理管线。原始数据集保持原尺寸。
基础 LeRobot ACT / ResNet18，chunk50，lr1e-5；每20k保存一次，因此包含60k/80k，
也会有20k/40k。该通用入口是新整理的基线，不声称与每个历史实验的超参数完全一致。
历史 ACT-HF 插件及原 trainer 同时保留于 `training/lerobot_policy_act_hf` 和
`training/scripts/lerobot_train_act_hf.py`，需要它时请显式区分。

本次整理没有启动 GPU 训练、没有改动当前云端队列。每个新数据集仍需跑真实 smoke。
模型部署时还要匹配标定、字段顺序、控制频率、temporal ensemble，不是加载权重就能直接安全执行。
