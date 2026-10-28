# 来源、许可与验证范围

Top 双目鱼眼标定入口调用外部仓库
`https://github.com/Back04oorer/fisheye_calibration`，固定 commit
`6ea4206ab2b1279171ee04a27990178b9992db57`。
源码保存在 Git 忽略的 `external/` 中，通过 `scripts/setup_calibration.sh` 获取；
本仓库提供 TUI 入口和原生 MJPEG 相机适配，复用上游标定、验证及导出流程。
`patches/fisheye-bottom-ab.patch` 保存 Bottom A/B 采集及验证目标、提示圈和回归测试的局部修改，
由安装脚本应用到上述固定版本；完整上游源码仍从外部仓库获取。
该版本未发现独立 LICENSE 文件，不将外部源码归入本仓库许可。
使用范围见[相机标定指南](06_camera_calibration.md)。

整理日期：2026-09-18。

- `src/zeno-wholebody-intervation/zeno-wholebody-robot`：当前本机 ros1_docker-main 工作代码，
  含最新 follower MIT 补偿适配器与 guard；没有覆盖回上游老版本。
- `src/piper_ros`：本机 Piper description/msgs，保留原 `LICENSE` 与模型网格。
- `workspaces/scripts`、CAN 脚本：当前启动/采集链路抽取，未执行其中硬件命令。
- `training`：Stereo-patch `26290bcd247f1fb373a517a7057f66fd40ef1198` 完整已跟踪源码快照，
  另补本机 ACT-HF trainer。保留原 Apache-2.0 LICENSE/NOTICE。不是训练数据或 checkpoint。
- `scripts/piper.py`、`scripts/train_act.py` 与分章指南：本次新增的独立通用入口。
- `scripts/setup_environment.py`、`scripts/setup_can.py`、`requirements/`、`docker/`：
  新增环境安装配方和四路 CAN 工具，不包含本机真实 USB 地址。
  训练快照的 `training/requirements.txt` 有一项集成修正：NumPy 从 `<2` 改为 `>=2.2,<2.3`，
  解决 LeRobot 0.4.4 间接依赖与 OpenCV 的约束冲突；来源哈希已同步更新。

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

### 安装配方补充验收（2026-09-18）

- 已从官方 `ros:noetic-ros-base-focal` 的固定 digest 构建新镜像（不是沿用本机自定义镜像）。
  实际执行 apt/Piper SDK 安装、10 个 catkin 包编译；初次构建发现缺少 `control_msgs`，
  已补入安装清单并重新构建成功。
- 使用 `--network none`、不映射 CAN/相机/GPU 的临时容器运行 `scripts/check_robot_env.py`：
  follower 驱动、补偿模块、ROS 消息与 SDK 导入成功，URDF 离线重力计算为有限数值。
  验证版本：Piper SDK 0.6.1、python-can 4.5.0、Pinocchio 3.3.0。
- 安装/CAN/通用入口共24项测试通过，训练契约10项、follower guard13项通过，
  共47项通过；主机8项依赖ROS的测试仍跳过。CAN单测使用模拟接口，不在实物上改接口状态。
- 用 uv 对 Python 3.10/Linux x86_64 的 `requirements/training.txt` 完成依赖解析。
  没有新装 GPU 训练环境，没有进行新的训练 smoke，也没有执行真实遥操或运动验收。

上述结果补充了初次整理的验证范围，仍不代表任意硬件配置无需检查即可直接运动。

GitHub Actions 配置保存在 `docs/ci/offline-checks.yml`，目前是未启用的模板。
上传账号缺少 `workflow` scope；使用具备该权限的账号将模板复制到
`.github/workflows/offline-checks.yml` 后可启用云端离线检查。本次验证在本机完成。
