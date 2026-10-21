# r3 科研审查

审查决定：**需修改并补充诊断（NEEDS_REVISION）；不接受冒烟通过，不开放 full 阶段。** 接受三份锁定权重已下载、逐块汇总算术和作者数据拟合复算结果。保留执行状态 FAILED；不是论文被证伪，也不是下载失败。本次仅审查和制定任务，没有启动新的 GPU 作业、下载或训练。

run_id：`parscale-20261021-smoke-r3`。event_id：`81cb5f2613797757a7df3799c66f696891fd3f95efc0d3c6709f3322f8f7f6a3`。report.json SHA256：`45f837dccbec7674856d1339e225309e8b43ca57923437e5111d6c4308066bb8`。

审查时未发现本事件已有 review.md。本记录对上述 event_id 有效；重复通知复用本审查，不重复启动任务。

## 身份、原始文件和版本

已按 workflow 的规范化 JSON 算法独立复算 event_id、task_sha256；completion.task 与本地固定 r3 task 逐字段一致。local-artifacts.json 的 event_id 一致，清单中的 22 个文件均按字节数和 SHA256 校验通过，拉取的 report.json 与嵌入报告一致。原始日志和 DSH 事件仅留在 ignored runs 目录，未上传。

独立审查脚本 `scripts/review_smoke.py` 在运行前提交；`review-checks.json` 保存核验与复算输出。评估源码与实际运行提交的版本相同，诊断源码与诊断运行提交的版本相同。审查未重新执行模型推理，也未重新跑优化器。

| 用途 | 实际提交 |
|---|---|
| r3 设计及 tiny synthetic prefix 初始化修复 | bc737786b7f0fec0aa2795bcd74dfa5d81d0a2ce |
| 固定权重 LFS 哈希及校验脚本 | f4195003f9265fc7c9890a5a65544ffbb5f8bd4c |
| CPU 拟合实际运行 HEAD | 247f74127000419ef75c328c614a3e0ffec2e957 |
| BF16 smoke 实际运行 HEAD | 27293651bcf6bfd156a177467ff8e099ecc3754a |
| FP32/BF16 cache diagnostics 实际运行 HEAD | a43288dcb8e5d1c03818d9a1dbb91798f19d3e67 |
| 执行者结果提交 | dec4aaa2d95ae2e0f7f23339d370cc71f9d09e3d |
| 独立审查脚本运行 HEAD / 审查写作前本地 HEAD | d0fb70c19e0bb6d1b3112c905156d4788e3d510f |

先前实现 `520c4bc`、拟合解析修复 `dc4b1cb`/`ea1ffde` 保留。远程快进 fetch 已完成，远程工作树干净，HEAD 为 dec4aaa。已用 private-repo.py 重新确认 `isPrivate=true` 并 relay push 到 d0fb70c；审查/后续任务的最终结果提交与推送回执另存接收目录 `delivery-receipt.json`，不能把它们当成上述推理运行 SHA。

## 可接受的证据及其边界

三份权重已在本次审查中通过 SSH 重新读取远程文件计算 SHA256，无权重下载到本机。结果与预先固定的 LFS 哈希、执行者 manifest 一致：

| 模型 | 文件字节数（manifest） | 本次独立远程 SHA256 |
|---|---:|---|
| P1 | 2687840768 | 0e6632e3e64c7125b67f0189d7c0d22d8fce73f1545c9ba37a3b683c0db7eeea |
| P2 | 2697379032 | c25a184a1240c3361311d58f2512a4856cb810b7c01d3960b4bd95c97be9c01e |
| P4 | 2706905312 | 8efe0426c53bdb8af3fe2d44847bc5533e05374fd32ef2bedaa14b274924c49d |

执行者 report.md 将 P4 字节数写成 2,690,905,312，属于文字笔误；机器清单和锁定配置均为 2,706,905,312。下载源不构成拒绝理由，用户已允许内容一致的可用源。r2 部分下载复用，r3 内 P1 超时后继续同一下载，未追加第四次准备尝试。

从 smoke.json 六条逐块 NLL 总和及 token_count 独立复算：每模型 2 个 validation block，每块 1023 个被评分 token，合计 2046。

| 模型 | NLL 总和 | token-weighted NLL | PPL |
|---|---:|---:|---:|
| P1 | 5448.640380859375 | 2.663069589863 | 14.3402402643 |
| P2 | 5303.748291015625 | 2.592252341650 | 13.3598286396 |
| P4 | 5262.222167968750 | 2.571956093826 | 13.0914074327 |

P2−P1 = −0.070817248213，P4−P1 = −0.091113496036，P4−P2 = −0.020296247823 nats/token。两块冒烟数据不能替代完整测试结果；未对它们作确认性 bootstrap 或科学效应判定。

代码检查显示：eval/inference_mode、每块重置 cache、logits[:-1] 对 targets[1:]、FP32 cross-entropy 求和、含空行的双换行拼接均按协议实现。manual/labels 最大差约 1.10e-7，记录中的因果检查差为 0，tiny P2 prefix/aggregation 梯度有限且非零。tiny 的 torch.empty prefix 已显式初始化，真实 checkpoint 不变。这些是已核验原始记录与源码，未独立重跑逐 token logits；原始 logits/targets 数组未交付，故从模型输出重新计算 CE 尚未验证。

三个 tokenizer 的 tokenizer.json 与 tokenizer_config.json 哈希、词表大小、probe 编码一致。但实现只检查 probe，未断言完整语料编码及词表映射一致。数据 manifest 记录 validation revision、fingerprint、text/token hash 和块 hash；本次未重建原始文本/token 流，其内容哈希只能作为待复验的 provenance。没有证据表明执行了 test 评分。

作者数据拟合：独立从固定 upstream/parametric_fit.py AST 提取全部 24 点，核对表格，再用报告参数重算预测、log-Huber objective 和 R²。Pile R² = 0.9986755850944447，Stack-v2-Python R² = 0.9978327010258554；两者均超过 .99。没有重跑 300 起点优化或独立训练；不能称为重新训练验证。上游 k 初始网格 .8 与论文附录 .9 的已知差异继续保留。

## 未通过的门槛与实现缺口

| 模型 | BF16 64-token max abs error | FP32 同序列 max abs error |
|---|---:|---:|
| P1 | 0.187500 | 0.00002193450927734375 |
| P2 | 0.140625 | 0.00002098083496093750 |
| P4 | 0.125000 | 0.00002479553222656250 |

BF16 三项均超过冻结阈值 .03，故保持 FAILED。FP32 同路径误差显著降低，支持精度差异解释，但不证明所有 BF16 差异都无害，也不能排除未覆盖输入上的 cache bug。16-token BF16 诊断是另一序列长度，不得混作主检查结果；report.metrics 中 BF16 数值来自该附加诊断。

DSH 日志中有一次 probe 的 `full_abs_max=21.875`，没有对应每个最坏误差坐标的 BF16/FP32 成对输出。不能从一个全局最大 logit 推导所有误差都仅为 1–2 ULP，也不能据此放宽阈值。FP32 diagnostic.data.token_ids_sha256 实际是父 1024-token block 的哈希，并非被比较的 64-token 序列；源码支持同一前缀，哈希字段需要修正。

执行资源规则发生实际偏离：日志显示 GPU 空闲约 19.04 GiB（已用 5078 MiB），低于协议 20 GiB；执行者仍继续，不能把硬门槛改称 guideline。preflight.json 的时间为 09:01:28Z，晚于推理开始，不足以作为每个模型开跑前记录。评估器没有每模型资源检查，也在 P1 cache 失败后继续 P2/P4；最终汇总检查替代了立即停止。NaN/失配也仅在末尾汇总，需修复。

full 实现还缺少 2000 次配对 block bootstrap、协议定义的 128-token prompt/32-token greedy generation 的 3 warmup+10 repeats 计时，以及明确的 full 输出/阶段门。现有 score_seconds 含标签 loss 等开销，不能当生成延迟或论文效率复现。

## 资源与运行结束核验

started.json：2026-10-21T08:21:42.967922Z；completion：09:03:10.316915Z。wrapper 墙钟 2487.346 秒（41.46 分钟），报告估计 2478.8 秒，两者保留。报告 GPU 0.06h 为估计，缺少连续活动积分，**未独立验证**。smoke 记录的最高 reserved=3.068359375 GiB 仅覆盖相应统计范围，不证明全部子进程 GPU 峰值或项目累计 GPUh。

本次 SSH 核查：parscale-smoke-r3.service 为 failed/exit-code，MainPID=0，ControlGroup 为空；RuntimeMaxSec=10800、KillMode=control-group。没有正在运行的该服务，无重复作业启动。wrapper 的 DSH exit_code=0 与 report FAILED 不矛盾：执行器返回报告，科研检查失败。远程独立 runtime 当前 du -sh 为 8.6G；该值不是累计磁盘写入量，也不包括所有目录。

项目累计 6 小时主动执行/4GPUh 不能由本次估计替代；后续须整合初始 smoke、取消的 r2、r3 和审查/修复耗时。额外 3 小时准备续跑额度已用于 r3，剩余额外准备重试为零。交付截止仍为 2026-10-22 北京时间 12:00。

## 后续任务及执行边界

已制定 `experiments/parscale-20261021-cache-diagnostic/task.json`，ready=true 表示设计已审查，execution_status 明确 NOT_STARTED。它是一次最多 1740 秒 wrapper/1800 秒 systemd、0.25GPUh 的 validation 诊断与实现修复，受项目剩余预算约束，不是新下载准备，也不是训练/full test。

任务固定两块 validation、两种 prefill 长度，要求保存实际序列/target hash、成对参考数值、BF16 与 FP32 target log-probability/NLL 差异；保留原 .03 结果，不自动放宽。补齐 tokenizer、立即停止、资源检查、bootstrap 和计时代码并做有意义的小样本检查。随后由 GPT 根据新证据明确决定是否需要版本化精度协议；只有门槛审查通过才可制定并运行固定 full 测试。

本次最新唤醒限于审查和制定任务，因此该任务未启动、未配置新运行监听。未来若实际启动长期任务，必须使用 systemd cgroup 限时，并核实真实 worker 与指向会话 `SESSION-REDACTED` 的 listener 都已启动，禁止 --last。常规 SSH/Git 权限已授权，无需重复询问。
