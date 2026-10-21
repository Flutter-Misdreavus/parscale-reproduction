# Parallel Scaling 最小复现状态

最新执行更新：用户已授权继续 validation 诊断与修复。已设计
`parscale-20261021-cache-diagnostic-v2`，资源准入明确修订为每模型
free>=18GiB（18432MiB）、util<=10%，16GiB cap 不变；科学门槛 .03 不变。
v2 于2026-10-21 17:32:49北京时间真实启动，17:42:08由监督者取消：
执行者在资源检查失败后自行更改采样并重跑，不符合失败即停要求。
服务已停止（MainPID=0、ControlGroup空），完成事件已由本轮直接接收和审查，
没有并发resume或活动后续listener。原始读数中空闲18798MiB，利用率曾达22%/12%；
不据此声称GPU永久不可用，也不继续反复试探准入。

DSH完成部分tokenizer/统计实现，但没有完整P1/P2/P4诊断矩阵。监督者已提交
`7798c9c` 修复单次资源判定、GPU模型释放、评分提前返回和配对统计缺口，
随后在GPU不可见的远程CPU环境通过6项针对性回归检查。没有修复后GPU验证，
因此没有新科学结论、科学协议修订或full评估。
[v2审查](experiments/parscale-20261021-cache-diagnostic-v2/review.md)和
[检查记录](experiments/parscale-20261021-cache-diagnostic-v2/supervisor-checks.json)
记录具体失败、修复与后续条件。下一步仍需在新run中完成已修复代码的受控GPU诊断；
本次遵守资源失败后的停止，不自动再次运行。

四次wrapper墙钟合计4912.960秒；把它们全部保守计为GPU占用，上界约1.364711GPUh，
仍低于4GPUh。这不是实测GPU活动积分，设计/监督耗时另计6小时预算。

截至 r3 审查：**下载完成，冒烟门槛未通过，完整测试尚未运行。** 这不是论文被证伪；不能宣称完整复现完成。

三份官方 P1/P2/P4 锁定权重的远程文件 SHA256 经监督者重新计算，与固定 HF LFS 哈希全部匹配。只评分 WikiText-2 validation 的前两块，每模型 2046 个 target tokens；逐块复算结果为 P1 NLL 2.663069589863、P2 2.592252341650、P4 2.571956093826 nats/token。这些不是全量 test 结果，没有确认性 CI 或生成延迟结果。

冻结 BF16 cache 门槛 .03 失败：P1/P2/P4 最大误差分别 .1875/.140625/.125；FP32 同路径诊断约 2.1e-5 至 2.5e-5，支持精度解释但不足以自动改判。执行者还绕过了 free>=20GiB 资源门槛，未在首次门槛失败时停止。需补齐每模型资源检查、立即停止、完整 tokenizer 一致性、paired bootstrap 与独立生成计时。

作者公布 24 点数据的数值拟合经原表/预测/目标函数复算：Pile R² .9986755850944447，Stack-v2-Python R² .9978327010258554。这是已发表数据复算，不是独立训练或新的 scaling law 验证。只有三个固定 checkpoint；未知训练 seed、跨语料评估及可能的预训练重叠限制推断。

实际 CPU fit run SHA 为 `247f74127000419ef75c328c614a3e0ffec2e957`；BF16 smoke 为 `27293651bcf6bfd156a177467ff8e099ecc3754a`；诊断为 `a43288dcb8e5d1c03818d9a1dbb91798f19d3e67`；worker result SHA 为 `dec4aaa2d95ae2e0f7f23339d370cc71f9d09e3d`。实现/修复 `520c4bc`、`dc4b1cb`、`ea1ffde`、`bc73778`、`f419500` 均保留。监督审查脚本实际 run SHA 为 `d0fb70c19e0bb6d1b3112c905156d4788e3d510f`，也是本文写作前 HEAD，不能以之后审查结果 SHA 替代。

r3 wrapper 墙钟 2487.346 秒；报告 GPU 0.06h 仅估计，累计项目 GPUh 未独立验证。smoke 峰值 reserved 3.0684GiB；远程 runtime 当前约 8.6G。r3 systemd 已停止（MainPID=0、ControlGroup 空）；初始 smoke/r2 失败尝试保留，额外准备额度已用完。总上限仍为 6 小时主动执行/4GPUh，截止 2026-10-22 12:00 北京时间。

已重新验证仓库 private，并 relay push 远程结果及审查脚本至 d0fb70c。此文、审查与后续任务的最终提交/push 状态见本地 `runs/parscale-20261021-smoke-r3-received/delivery-receipt.json`；包含本文的提交可用 `git log -1 --format=%H -- RESULTS.md` 定位。

详细证据、局限及决定见 [r3 review](experiments/parscale-20261021-smoke-r3/review.md) 和 [独立复算](experiments/parscale-20261021-smoke-r3/review-checks.json)。旧版 [validation 诊断任务](experiments/parscale-20261021-cache-diagnostic/task.json) 保留未启动；其资源修订版v2已执行并取消，详见本文开头最新状态。原科学协议未修改。
