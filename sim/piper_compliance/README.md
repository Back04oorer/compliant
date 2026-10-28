# Piper shared simulation utilities

旧版 MuJoCo / Stable-Baselines3 PPO（含 CMA v1–v3、教师预训练和旧交互页面）已移除。
当前 RL 训练使用 [Piper SoftMimic](../piper_softmimic/README.md)。

本目录只保留 SoftMimic 仍使用的工具：

- `model.py`：控制 URDF 与 MuJoCo 模型的一致性校验。
- `kinematics.py`：锁定夹爪的模型加载、固定参考姿态与 IK，用于增强数据和评估目标生成。
- `env.py`：冻结的 SoftMimic 评估入口所需的 IK 兼容接口，不含旧 Gym/RL 环境。
- `dashboard.py`、`training_curves.py`、`web/`：SoftMimic 训练曲线、评估回放和 Isaac 实时对比代理。

启动网页服务（不依赖旧实验输出）：

```bash
.venv-softmimic/bin/python -m sim.piper_compliance.dashboard --host 127.0.0.1 --port 8765
```

`requirements.txt` 只列出这些共享工具的依赖；Isaac/SoftMimic 环境见上面的适配说明。
历史输出、数据和权重保留在 `outputs/piper_compliance/`，不再提供旧 PPO 的训练或加载入口。
