# 4. ACT 训练

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

```bash
python scripts/train_act.py --dataset /data/lerobot/my_task_left \
  --output /data/train/rgb_act_smoke --arm-mode single --smoke
python scripts/train_act.py --dataset /data/lerobot/my_task_left \
  --output /data/train/rgb_act --arm-mode single --batch-size 16 --steps 80000
```

入口按 meta/info.json 显式选择 Top + 活动腕/双腕以及 state，不把 depth 当成 RGB 图。
基础 LeRobot ACT / ResNet18，chunk50，lr1e-5；每20k保存一次，因此包含60k/80k，
也会有20k/40k。该通用入口是新整理的基线，不声称与每个历史实验的超参数完全一致。
历史 ACT-HF 插件及原 trainer 同时保留于 `training/lerobot_policy_act_hf` 和
`training/scripts/lerobot_train_act_hf.py`，需要它时请显式区分。

本次整理没有启动 GPU 训练、没有改动当前云端队列。每个新数据集仍需跑真实 smoke。
模型部署时还要匹配标定、字段顺序、控制频率、temporal ensemble，不是加载权重就能直接安全执行。
