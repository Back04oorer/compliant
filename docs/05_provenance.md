# 来源、许可与验证范围

整理日期：2026-09-18。

- `src/zeno-wholebody-intervation/zeno-wholebody-robot`：当前本机 ros1_docker-main 工作代码，
  含最新 follower MIT 补偿适配器与 guard；没有覆盖回上游老版本。
- `src/piper_ros`：本机 Piper description/msgs，保留原 `LICENSE` 与模型网格。
- `workspaces/scripts`、CAN 脚本：当前启动/采集链路抽取，未执行其中硬件命令。
- `training`：Stereo-patch `26290bcd247f1fb373a517a7057f66fd40ef1198` 完整已跟踪源码快照，
  另补本机 ACT-HF trainer。保留原 Apache-2.0 LICENSE/NOTICE。不是训练数据或 checkpoint。
- `scripts/piper.py`、`scripts/train_act.py` 与分章指南：本次新增的独立通用入口。

`SOURCE_MANIFEST.json` 记录源码快照各文件 SHA256，以便之后与运行目录对照。
部分继承的脚本保留历史参考绝对路径，不应直接当作别的机器配置执行。
推荐入口改为仓库相对路径/显式外部资源；硬件地址在不提交的 local config 中维护。

## 许可证边界

不是整库 Apache 或 MIT。子目录保留已有许可证；部分 Zeno/ROS package.xml 标注 TODO，
需要原作者/权利人确认后才能决定公开发布与再分发许可。当前仓库设为私有。
不能通过本次整理授予第三方源码、基础模型权重的额外权利。

## 验证边界

已做离线源码/launch XML/参数引用/脚本语法与针对性单元测试、敏感信息和大文件检查。
不会启动驱动、重力补偿、遥操、CAN、相机或训练。
另在本机 `ros1-noetic-local` 镜像的临时容器内完成 **10 个 catkin 包的全新工作空间构建**，
容器禁用网络、不映射 CAN/USB/GPU，源码只读挂载、在临时副本编译，退出即删除容器。
入口测试7项、训练契约测试10项、follower guard测试13项通过；8项依赖ROS的测试在主机跳过。
这不等于在空白 Ubuntu 上从零安装全部依赖，也没有对新入口重新做硬件运动或 GPU smoke 验收。
因此这是整理后的源码与可验证入口，不是已在所有机器即插即用的认证系统。

GitHub Actions 配置保存在 `docs/ci/offline-checks.yml`，目前是未启用的模板。
上传账号缺少 `workflow` scope；使用具备该权限的账号将模板复制到
`.github/workflows/offline-checks.yml` 后可启用云端离线检查。本次验证在本机完成。
