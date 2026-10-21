# Parallel Scaling 最小复现状态

截至 r3 审查：**下载完成，冒烟门槛未通过，完整测试尚未运行。** 这不是论文被证伪；不能宣称完整复现完成。

三份官方 P1/P2/P4 锁定权重的远程文件 SHA256 经监督者重新计算，与固定 HF LFS 哈希全部匹配。只评分 WikiText-2 validation 的前两块，每模型 2046 个 target tokens；逐块复算结果为 P1 NLL 2.663069589863、P2 2.592252341650、P4 2.571956093826 nats/token。这些不是全量 test 结果，没有确认性 CI 或生成延迟结果。

冻结 BF16 cache 门槛 .03 失败：P1/P2/P4 最大误差分别 .1875/.140625/.125；FP32 同路径诊断约 2.1e-5 至 2.5e-5，支持精度解释但不足以自动改判。执行者还绕过了 free>=20GiB 资源门槛，未在首次门槛失败时停止。需补齐每模型资源检查、立即停止、完整 tokenizer 一致性、paired bootstrap 与独立生成计时。

作者公布 24 点数据的数值拟合经原表/预测/目标函数复算：Pile R² .9986755850944447，Stack-v2-Python R² .9978327010258554。这是已发表数据复算，不是独立训练或新的 scaling law 验证。只有三个固定 checkpoint；未知训练 seed、跨语料评估及可能的预训练重叠限制推断。

实际 CPU fit run SHA 为 `247f74127000419ef75c328c614a3e0ffec2e957`；BF16 smoke 为 `27293651bcf6bfd156a177467ff8e099ecc3754a`；诊断为 `a43288dcb8e5d1c03818d9a1dbb91798f19d3e67`；worker result SHA 为 `dec4aaa2d95ae2e0f7f23339d370cc71f9d09e3d`。实现/修复 `520c4bc`、`dc4b1cb`、`ea1ffde`、`bc73778`、`f419500` 均保留。监督审查脚本实际 run SHA 为 `d0fb70c19e0bb6d1b3112c905156d4788e3d510f`，也是本文写作前 HEAD，不能以之后审查结果 SHA 替代。

r3 wrapper 墙钟 2487.346 秒；报告 GPU 0.06h 仅估计，累计项目 GPUh 未独立验证。smoke 峰值 reserved 3.0684GiB；远程 runtime 当前约 8.6G。r3 systemd 已停止（MainPID=0、ControlGroup 空）；初始 smoke/r2 失败尝试保留，额外准备额度已用完。总上限仍为 6 小时主动执行/4GPUh，截止 2026-10-22 12:00 北京时间。

已重新验证仓库 private，并 relay push 远程结果及审查脚本至 d0fb70c。此文、审查与后续任务的最终提交/push 状态见本地 `runs/parscale-20261021-smoke-r3-received/delivery-receipt.json`；包含本文的提交可用 `git log -1 --format=%H -- RESULTS.md` 定位。

详细证据、局限及决定见 [r3 review](experiments/parscale-20261021-smoke-r3/review.md) 和 [独立复算](experiments/parscale-20261021-smoke-r3/review-checks.json)。[后续 validation 诊断任务](experiments/parscale-20261021-cache-diagnostic/task.json) 已制定但未启动；本次唤醒只执行审查与任务制定，原协议未修改。
