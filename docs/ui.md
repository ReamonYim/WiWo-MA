# Browser workflow

The current UI uses Chinese and bilingual controls. The following map matches
the actual interface; the documentation does not invent an English-only UI.

| Task | Current control |
|---|---|
| Select panel | 选择面板 |
| Target coordinate frame | 目标输入坐标 |
| Submit target pose | 执行目标姿态 |
| Queue selected panel | 将所选目标加入联合动作 |
| Submit joint action atomically | 原子执行全部联合动作 |
| Start / pause / continue | Start / 开始预览; Pause / 暂停; Resume / 继续 |
| Save / restore mechanics | 保存机械快照; 恢复机械快照 |
| Load recorded HDF5 | 加载真实保存数据 |
| Seek selected recorded time | 跳到时间轴所选帧 |

Choose a physical preset or edit the configuration forms. Applying a new
configuration validates it, resets the mechanical timeline and invalidates
stale CSI. Select a panel, inspect its actual constraints, then enter an
allowed target. For `constrained_BS_local`, translation and rotation-vector
targets are relative to the configured initial mount, not the last position.
Gizmos are intended for free 6DMA world-pose inputs; restricted mechanisms use
the explicit permitted axes.

Input edits create a draft. Submit the target and inspect the actual payload
and its state: submitted/waiting for a control tick, executing, completed,
superseded or no-op. Start advances bounded mechanics; it does not itself
submit the typed target. An already reached target has no new displacement.
Incomplete or nonfinite numbers are explicitly refused before submission.

For a joint action, configure and queue each panel, then submit the queue.
Check all target poses before execution. Workspace, speed, acceleration,
rotation-axis and collision checks are not bypassed by the UI.

The first connected session holds the operator lease. Other browsers observe
the shared state. The operator may release control for another session to
claim; disconnecting does not terminate a separately running job. This is a
local research coordination mechanism, not internet multi-tenant security.

## Read the mode label

- **Mechanical preview:** actual integrated geometry, no newly computed CSI.
- **Native background/current-pose job:** asynchronous SDK work, with actual
  submitted frames and waiting/completed/failed status.
- **Stored replay:** previously generated CSI and geometry at recorded times.

Do not interpret a drawing frame, timeline update or old CSI display as a new
ray-tracing query. Load a closed HDF5 for playback, then use Start/Pause,
frame stepping and the timeline. Numerical validation labels come from bound
evidence; playback never promotes an unchecked sample to Passed.
