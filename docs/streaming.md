# 异步专家层搬运

`store/async_streamer.py` 沿用同步版的原字节 bank 和句柄接口。通过 `build_model(..., streaming="async")` 选择；默认仍为同步基线。异步要求 CUDA 和由 Torch 分配且持续存活的 pinned CPU bank。

每个 GPU 槽记录层号、generation、pins、copy_done 和 last_reader。实际 acquire 的层号是唯一依据：计算流先等待该 generation 的拷贝完成；release 把最后读操作的事件排入计算流。槽即使 Python pins 为0，拷贝流覆盖前仍等待 last_reader。每次覆盖新建事件并增加generation，避免重复记录事件和过期句柄混淆。

prefetch 仅在有可覆盖槽时发起，不等待。预测错误会增加流量；命中、占用检查和 acquire 的实际层号仍保证正确结果。槽 Tensor 对计算流和拷贝流登记 allocator 生命周期；CPU bank 由 streamer 持有，结束时调用 `close()` 完成传输和读者后再释放。未释放句柄、跨计算流访问、重复释放和关闭后 acquire 均拒绝。

训练前向设置 `phase="FORWARD"`，反向前设 `"BACKWARD"`；非重入重算可能改变实际访问顺序，事件正确性不依赖提示。评测同样明确使用 FORWARD。专家 autograd 不保存层句柄，反向重新 acquire。

`collect_timings()` 只回收已完成 CUDA 事件，训练每step调用时不强制逐层同步；报告/测试可显式要求同步。统计分别记录拷贝GPU时间、acquire等待GPU时间、入队CPU时间、miss/hit/预取/字节。同步版的 copy_wall_seconds 在异步版恒为0，应使用异步专用计时字段。

验证入口：

```powershell
python -B tools/run_recorded.py --output <新运行目录> -- tools/check_async.py --experts <真实bundle>/experts --output '{run}/results'
python -B tools/run_recorded.py --output <另一新运行目录> -- tools/check_tiny_training.py --bundle <tinybundle> --streaming async --steps 100 --output '{run}' --commit <完整提交>
```

真实P3全部通过，随后完整33项回归与异步100步/发布/续训通过，且100步与同步基线全部E2。M1逐项审核由 `tools/audit_m1.py` 读取真实run、数值/状态证据与源码摘要；原始运行、大型快照、trace和权重留在仓库外，小型摘要见 `docs/evidence`。

Profiler的跨流重叠使用人为延迟的读取压力测试。真实训练是否提速必须在真实训练中另测；tiny及传输测试的带宽不能代替整模训练吞吐。
