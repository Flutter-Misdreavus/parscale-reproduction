# Parallel Scaling 最小复现状态

2026-10-21 v3审查完成：**下载已完成；单次修复后GPU诊断在资源准入处停止；full尚未运行，不能宣称最小全量复现完成。**

`parscale-20261021-cache-diagnostic-v3` 在P2/BF16前仅一次读数为free18720MiB/util12%，利用率超过固定10%上限，立即停止，无重试。GPU run SHA为`1f1aeedb5d453a112e94db46fbb6de331f5a7933`；本次已使用唯一授权尝试，未改BF16 .03门槛，未继续full。

监督者在CUDA不可见环境独立读取P1原始数组：2046个validation target的FP32 NLL为2.663731906636；BF16保存target logprobs复算为2.663069842718，差-0.000662063918 nats/token。FP32 cache最大误差2.2888e-5，BF16为0.203125，仍未通过.03。P2只有FP32原始输出、P4缺失，实际配对bootstrap及计时未完成。平均NLL接近不能替代完整数值验证。

[本次审查](experiments/parscale-20261021-cache-diagnostic-v3/review.md)记录证据范围与局限；[CPU独立结果](experiments/parscale-20261021-cache-diagnostic-v3/supervisor-independent.json)运行代码SHA为`4a5011c`。修复SHA`7798c9c`已在v2后通过6项CPU回归，本次P1运行部分验证了修复；不能把部分完成写成完整诊断通过。

远程worker结果SHA为`0d7fd01`，其中包含原始诊断日志，因此保留在远程且没有合并/推送该历史；脱敏结果relay提交为`4a07224`。最终审查HEAD和private验证/push状态记录于本地`runs/parscale-20261021-cache-diagnostic-v3-received/delivery-receipt.json`。包含本文的审查提交可用`git log -1 --format=%H -- RESULTS.md`定位。

诊断墙钟130.764s，wrapper258.125s；五次已登记wrapper合计5171.085s，保守GPU占用上界约1.436413h，报告活动GPUh仅估计。主动墙钟没有完整积分，不能用wrapper时间代替6h总预算。仍受4GPUh/6h及今日21:30外层停止时间限制。

最新服务核查MainPID=0、ControlGroup空，无本项目worker或listener；本轮直接接收已结束，无并发resume。当前决定为证据不足而停止，不自动新增GPU任务。v2失败、早期两次准备和r3失败全部保留；不得把资源停止解释为论文科学失败。

## r3及此前结果（历史记录，以下提交与资源数不代表最新HEAD）

截至 r3 审查：**下载完成，冒烟门槛未通过，完整测试尚未运行。** 这不是论文被证伪；不能宣称完整复现完成。

三份官方 P1/P2/P4 锁定权重的远程文件 SHA256 经监督者重新计算，与固定 HF LFS 哈希全部匹配。只评分 WikiText-2 validation 的前两块，每模型 2046 个 target tokens；逐块复算结果为 P1 NLL 2.663069589863、P2 2.592252341650、P4 2.571956093826 nats/token。这些不是全量 test 结果，没有确认性 CI 或生成延迟结果。

冻结 BF16 cache 门槛 .03 失败：P1/P2/P4 最大误差分别 .1875/.140625/.125；FP32 同路径诊断约 2.1e-5 至 2.5e-5，支持精度解释但不足以自动改判。执行者还绕过了 free>=20GiB 资源门槛，未在首次门槛失败时停止。需补齐每模型资源检查、立即停止、完整 tokenizer 一致性、paired bootstrap 与独立生成计时。

作者公布 24 点数据的数值拟合经原表/预测/目标函数复算：Pile R² .9986755850944447，Stack-v2-Python R² .9978327010258554。这是已发表数据复算，不是独立训练或新的 scaling law 验证。只有三个固定 checkpoint；未知训练 seed、跨语料评估及可能的预训练重叠限制推断。

实际 CPU fit run SHA 为 `247f74127000419ef75c328c614a3e0ffec2e957`；BF16 smoke 为 `27293651bcf6bfd156a177467ff8e099ecc3754a`；诊断为 `a43288dcb8e5d1c03818d9a1dbb91798f19d3e67`；worker result SHA 为 `dec4aaa2d95ae2e0f7f23339d370cc71f9d09e3d`。实现/修复 `520c4bc`、`dc4b1cb`、`ea1ffde`、`bc73778`、`f419500` 均保留。监督审查脚本实际 run SHA 为 `d0fb70c19e0bb6d1b3112c905156d4788e3d510f`，也是本文写作前 HEAD，不能以之后审查结果 SHA 替代。

r3 wrapper 墙钟 2487.346 秒；报告 GPU 0.06h 仅估计，累计项目 GPUh 未独立验证。smoke 峰值 reserved 3.0684GiB；远程 runtime 当前约 8.6G。r3 systemd 已停止（MainPID=0、ControlGroup 空）；初始 smoke/r2 失败尝试保留，额外准备额度已用完。总上限仍为 6 小时主动执行/4GPUh，截止 2026-10-22 12:00 北京时间。

已重新验证仓库 private，并 relay push 远程结果及审查脚本至 d0fb70c。此文、审查与后续任务的最终提交/push 状态见本地 `runs/parscale-20261021-smoke-r3-received/delivery-receipt.json`；包含本文的提交可用 `git log -1 --format=%H -- RESULTS.md` 定位。

详细证据、局限及决定见 [r3 review](experiments/parscale-20261021-smoke-r3/review.md) 和 [独立复算](experiments/parscale-20261021-smoke-r3/review-checks.json)。旧版 [validation 诊断任务](experiments/parscale-20261021-cache-diagnostic/task.json) 保留未启动；其资源修订版v2已执行并取消，详见本文开头最新状态。原科学协议未修改。
