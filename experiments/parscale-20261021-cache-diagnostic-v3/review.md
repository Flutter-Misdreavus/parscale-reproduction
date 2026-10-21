# v3 独立审查：接受部分证据，诊断未完成，不进入 full

审查日期：2026-10-21，北京时间。单次授权已使用，无再次 GPU 尝试、训练或下载。

## 身份和执行

completion event `fa39eed8a41c7cf3bee01d022883ccc5efa826d450d3ac65fd14fbec748cd2ee`，冻结任务 canonical SHA256 `7aa19146827d8ea351466e0fffd6ff0110163d9861e596f6d5bfc5e6fb61cfe5`：使用 workflow.validate_event 与本地 task 全量核对通过。local-artifacts 的 event_id 一致，14 个接收文件的长度与 SHA256 全部匹配；report SHA256 `df5294dcb21c940f64cab67ef83e30d99f80b9fd9b0c72353407e9bfd9a4421c`。此前本事件无 review，未重复启动。

实际 GPU run SHA `1f1aeedb5d453a112e94db46fbb6de331f5a7933`，继承修复 `7798c9c` 和 v2 审查 `e620326`。真实命令 `bash scripts/run_diagnostic_once.sh runs/parscale-20261021-cache-diagnostic-v3`；持久 once-only 标记、895s timeout 加5s终止宽限、wrapper1740s/systemd1800s。没有更改 BF16 .03 科学门槛。P1 FP32/BF16 和 P2 FP32 执行后，于10:29:55.619614 UTC在 P2/BF16 前单次检查得到 free18720MiB、util12%，前者满足18432，后者超过10%，立即停止。每模型检查只有一个样本，未重试。无法从单个读数判断12%来自外部任务还是自身残留采样；不作归因。

completion BLOCKED 与日志异常一致。wrapper exit_code0 只说明 DSH 完成报告，不能解释为实验成功。systemd MainPID0、ControlGroup空、Result=exit-code；SSH进程检查未发现本项目 worker，直接 artifact receiver 已退出，无活动 listener 或并发 resume。

## 原始证据独立复算

CPU 审查代码实际 SHA `4a5011cc4d77cd520559c43c26d7e12be0328d6d`，在独立远程 worktree 运行 `scripts/review_diagnostic_raw.py --allow-partial`，CUDA_VISIBLE_DEVICES为空、HF离线。没有模型推理或新 GPU 运行。结果见 supervisor-independent.json；该文件接收 SHA256 `6d8131b91cefb4a7a9c0e8abbdb8502ddfc8eaf7a42084dc5acc67471268cdb1`。

重新从锁定缓存 validation 文本构造262337个token，全文和token哈希匹配。P1六个原始数组文件长度/哈希全部核验，并从两块1024token的FP32 logits以float64 logsumexp重建2046个target的NLL。BF16整块原始logits未保存，故其NLL只从已保存target log probabilities重加权；不声称从BF16整块logits独立重建。两个64token序列的FP32/BF16原始full/cached数组均在远程读取，核验token位置、两种prefill(1/32)的cache误差、跨精度logits差异和target logprob差异。

| P1 指标 | 独立结果 |
|---|---:|
| FP32 token-weighted NLL，从原始logits重建 | 2.663731906636 nats/token |
| BF16 token-weighted NLL，从保存target logprobs复算 | 2.663069842718 nats/token |
| BF16 − FP32 | -0.000662063918 nats/token |
| FP32最大full/cached绝对误差 | 0.00002288818359375 |
| BF16最大full/cached绝对误差 | 0.203125 |

独立FP32 NLL与worker FP32 reduction结果差约7.74e-7 nats/token；BF16重新求和差约1.34e-7，符合浮点归约精度差异，不修改科学阈值。跨精度target logprob最大差0.11033678也得到数组复核；平均NLL小差异不能排除单token显著误差。P1结果支持有限精度影响的解释，但未证明所有路径没有cache问题。原门槛仍失败，不以观察到的最大值反设容差。

P2的四个FP32原始文件额外通过远程sha256sum核对清单，但未独立重建其统计，不能算作完整P2配对证据。P2 BF16、P4、生成计时及实际配对bootstrap缺失。analysis-checks只含合成fixture的通过结果，actual_validation_two_block_bootstrap为null，不是数据置信区间。tokenizer清单报告三个模型的文件/词表/全validation编码一致；独立审查重建了P1全文编码，其余一致性依据已审查实现和保存清单，未重新运行三个tokenizer。没有full test指标、CI或速度结论。

## 版本、资源和数据保留

远程worker结果 `0d7fd010c447ba5acfdda9f2b4e7ae4fa57d12a4` 含原始 diagnostic.log，违反PROTOCOL中raw logs不入Git的规则（报告deviations=[]遗漏此事）。为防止上传，该提交没有合并进relay分支。原始远程历史保留，11份小型指标/清单导出为脱敏relay提交 `4a07224`，排除了日志。无数据、权重或DSH日志上传。远程主checkout仍为0d7fd01；CPU审查worktree固定4a5011c。这些不同SHA有明确用途，不能混作GPU运行SHA。

本次诊断命令墙钟130.764s（作为GPU上界0.036324h），wrapper258.125s。报告0.012617GPUh为间接估计，非独立活动积分。已知五次wrapper总计5171.085s，全部保守计为GPU占用约1.436413h；此界仅覆盖已登记wrapper，不声称覆盖独立CPU诊断或未记录阶段。项目主动墙钟未完整计量，不以wrapper总和代替6h预算；外层今日21:30停止仍有效。报告峰值allocated3.671GiB、reserved4.018GiB，低于16GiB cap，来自已保存阶段；中断P2的精确峰值未证实。原始诊断数组约3.189GB留远程。

## 决定与下一步

接受“遵守单次资源停止且P1部分证据有效”的执行结论；不接受完整诊断成功或full阶段通过。当前证据仍不足，按最新授权停止，不另造重试，不修改科学协议，不启动full。主要阻塞为P2/BF16资源准入失败、完整P2/P4配对诊断与计时缺失，以及原BF16门槛仍失败。这不是论文被证伪，也不是最小全量验证已完成。

若上层今后另行决定恢复，需先明确剩余预算与新的固定任务，补齐缺失诊断并基于独立证据预先制定科学协议；现有v3授权不包含再次GPU尝试。full实现仍需按新协议冻结并验证，不能直接启用当前被封锁的旧full入口。交付范围现为作者数据拟合复算、validation冒烟及部分数值诊断。最终审查HEAD/push状态见本事件delivery-receipt.json；包含本文的提交用git log -1 --format=%H -- experiments/parscale-20261021-cache-diagnostic-v3/review.md定位。
