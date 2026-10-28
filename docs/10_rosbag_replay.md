# 按录制规则回放 rosbag 轨迹

主菜单 `b. Replay rosbag on robot (local ARM)` 默认使用 `--rules recording`。
读取 episode 同目录的 `repro_snapshot.json`，以录制时的 MIT 控制规则回放关节目标。
向 `/robot/arm_left/vla_joint_cmd` 和 `/robot/arm_right/vla_joint_cmd` 发布数值，
经 mux 输出到同一套 follower 驱动。遥操、ACT 与回放共用驱动，切模式无需重启驱动。
不加载 policy，不做归一化，不运行整包 `rosbag play`，不发布 bag 中的模式、使能或 mux 消息。

## 启动

```bash
docker exec -it -w /opt/piper piper-control python3 scripts/teleop_tui.py --execute
```

1. 退出上一轮 ACT 或回放执行桥，保持现有从臂驱动运行；两者不能同时占用 policy 输入。
2. 若从臂驱动尚未启动，在另一终端选主菜单 `2` → `1`，保持该终端运行。
   此入口会使能 follower，包含重力补偿与默认选择 teleop 的 mux。选项 9 已移除。
   无需手动执行 `2` → `8`；leader 和相机不是回放的前置条件。
3. 用于回放的 TUI 在主菜单选 `b`，从列表选 `.bag`，可 `/` 筛选、浏览目录或 Tab 手输路径；
   默认高亮最近修改的完整 `.bag`。来源默认 `mux`，这里只表示读取 bag 内
   录制的命令数值；`state` 表示读取录制的实测位置，两者都通过 policy 输入与 mux 回放。
   来源和模式用方向键选择，Enter 确认；模式为 `robot`、`dry-run` 或只读的 `check`。
   路径、时长、速度输错均留在当前项修改；Esc 取消返回，尚未启动回放。
   按键说明见 [TUI 统一选择与输入](01_startup.md#tui-selection-and-input)。
4. 输入起点、录制时长、倍速。默认 `0`、`all`、`1`，播放当前 episode 的完整公共时间段。
   `all` 指所选 bag 的整段，不自动串播其他 episode。也可输入秒数；1 倍速的 10 秒片段执行约 10 秒；
   起点定位单独计时。倍速只按用户输入缩放时间，没有额外的自动降速。
5. 就绪后，在同一终端输入 `ARM` 并回车。反馈和连接检查通过后，先发送实测位置保持指令、将两臂 mux 切到 policy 并确认，
   再慢速定位到片段起点，按录制规则回放。
   **空格暂停/继续**，暂停保持当前目标，暂停时间不消耗轨迹时长。
   **Enter 慢速回到当前 episode 第 0 帧并保持**，再按空格从头播放。
   播完保持末帧，仍可按 Enter 返回起点。`q` 或 Ctrl-C 退出并恢复 teleop 输入；退出不会断电。
   暂停、复位和末帧保持期间仍由 policy 控制；恢复遥操前应先对齐 leader，恢复后从臂会跟随 leader。
   这些按键无需额外回车；首次启动仍需输入 `ARM` 再回车。定位过程中也可以暂停。
   若手动选择了非零起点，Enter 仍返回该 bag 第 0 帧；空格后先慢速到所选片段起点再播放该片段。

若 policy 输入存在其他命令发布者、mux 输出或状态发布者不正确，或驱动没有订阅 mux 输出，会拒绝 ARM。
运行期间也检查 mux 选择、命令发布者和驱动连接；反馈失效或手动切走 mux 时停止回放。
切换部分失败会撤销已尝试的接管；停止/退出时恢复仍由执行桥控制的臂至 teleop，尊重手动切到其他来源。
恢复失败会打印并记录错误，可用 `2` → `7` 手动恢复遥操。进程被强杀时不能自动恢复 mux。
启动驱动的 CLI 等价命令：

```bash
docker exec -it -w /opt/piper piper-control python3 scripts/piper.py robot --execute
```

CLI 示例（完整 `.bag` 路径或容器内相对路径均可）：

```bash
docker exec -it -w /opt/piper piper-control /opt/piper-inference/bin/python scripts/replay_rosbag.py \
  --bag rosbag_data/experiment1/episode_001/episode.bag \
  --source mux --mode robot --speed 1 \
  --rules recording --output-dir outputs/rosbag_replays/experiment1_full_replay
```

省略 `--duration` 默认播放全集；`--duration 10` 可只选 10 秒。
每次使用新的输出目录。改成 `--mode check` 可只读检查，不要求驱动、ROS master 或真实硬件。

## 默认录制规则

- 快照中必须有两臂完整的 MIT 位置控制参数。回放开始前和运行期间核对当前 ROS 参数：
  CAN 角色、控制/发布频率、MIT kp/kd、speed、输入通道、滤波、重力补偿及夹爪映射。
  参数不一致会报出具体路径；回放不会更改驱动参数、零位或硬件限位。
- experiment1 的录制命令约 100 Hz，驱动控制循环 50 Hz。回放按录制发布频率同步重采样，
  整段预加载，由执行桥按经过时间选择目标，不依赖 ACT 的 50 点分块上传。
  这是按原时序采样关节轨迹，不保证逐条 ROS 消息或机械运动逐毫秒完全相同。
- 关节目标遵从原 MIT 驱动的协议输入范围 ±12.5 rad，夹爪使用原映射和驱动限幅。
  不叠加 ACT/URDF 逐关节范围、0.2 rad/s 的 ACT 轨迹速度上限、加速度整形或平滑。
  J2 负值、J3 正值、J5 超过 70° 不再仅因 ACT/JointCtrl 参考范围被拒绝或修改。
  这些参考范围的越界值仍会列在预检报告中。
- 回放段不使用录制时没有的 ACT 跟踪误差截止；experiment1 原始反馈已有约 0.31 rad
  的命令/反馈分离。起点定位保留慢速过渡、静止条件和超时检查。
- 原始 bag 保持不变。仍拒绝非有限值、无效关节名字/排列、乱序/重复时间戳、
  超过 0.1 秒的数据缺口及超出 MIT 输入范围的数据。左右取公共时间区间，不外推缺失数据。
- 仍要求本地 ARM、当前从臂反馈与兼容的驱动/CAN 诊断、使能状态、无驱动故障、
  原有 follower 重力补偿、独占 policy 命令发布者、mux 的 policy 选择，以及驱动关节、夹爪均订阅对应的 `joint_cmd_mux`。
  反馈过期、控制循环超时、客户端失联或 mux 被接管会停止，暂停/末帧保持期间也会检查。
  单次轨迹最多 1800 秒，定位最多 60 秒；暂停时间不计入运动预算。

录制兼容模式也不会新增录制时没有的逐关节机械限位保护。它复现的是原控制规则；
不做避障、场景复原或零位校准。实机验证仍需现场 ARM。

## experiment1 的原因

录制快照证明两臂使用 MIT，mux 输入是对应 leader 的 `joint_states_single`。
基础驱动 `_mit_control` 只约束 MIT 协议编码范围，SDK 的 `JointMitCtrl` 也不调用
普通 `JointCtrl` 的逐关节限位函数。bag 中的 leader、mux 和 follower 实测流都出现了
此前被 ACT 拒绝的数值，问题来自两条路径使用了不同规则，并非回放读错、排序或归一化。

当前固件 `S-V1.8-2` 的 0x473 速度单位是 0.01 rad/s，左右查询读数是 2、3 rad/s。
旧诊断按老版本单位读成 0.2、0.3；该值不是录制 MIT 路径的回放限速依据。
协议依据见 [SDK changelog](https://github.com/agilexrobotics/piper_sdk/blob/master/CHANGELOG.MD)。

## 输出和独立的 ACT 规则模式

`request.json` 保存参数；`preflight.json` 保存检查、参考范围越界及实际回放时长；
`recorded_trajectory.json` 保存原始序列；`replay_targets.json` 保存重采样目标；
`recording_plan.json` 保存执行桥使用的完整轨迹和快照参数；`reset_pose.json` 保存起点。
执行桥的私有 session 和事件记录在 `bridge/`。故障停止返回失败，不会显示为完成。

若明确需要以前的受限轨迹变体，使用 `--rules act --constrain`，才会按 ACT 限位裁剪、
重新计时和平滑。该变体改变轨迹，与默认的录制兼容回放是两个明确选择。
该 ACT 规则变体仍用 Enter 停止。普通 RGB ACT 保留它的 MIT J2/J3 规则、其他关节限位、速度、跟踪与观测检查；
持续推理没有 60 秒截止，首帧定位仍有 60 秒超时。
