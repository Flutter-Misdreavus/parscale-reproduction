# cache-diagnostic-v2 审查与停止记录

决定：**NEEDS_REVISION / GPU 阶段已停止，full 阶段未获证据支持。**
本轮确实派发并运行远程 DSH，不是只设计任务。监督者按任务停止条件结束
systemd 整个 cgroup；不把取消当科学负结果，不修改 r3 的 BF16 .03 失败。

事件 `726abfddccd892a5d27e8bb1e06c336c11b7ea26e73c4506515baaced6609e1f`，
固定 task SHA256 `1ae7b2defa2c873898745f47280ebd706b23fe8c2a67896ef1c2de3977fe93e4`。
completion 状态 CANCELLED，时间 2026-10-21T09:42:08.862952Z，wrapper559.386秒。
wrapper 因没有最终 report.json 生成降级 FAILED 报告；不是成功科学报告。
已独立核验 event_id、固定 task 和 local-artifacts 的事件身份、三个原始文件的
大小/SHA256。没有已有本事件审查或重复启动。

## 实际执行及失败依据

2026-10-21T09:32:49.475019Z 启动，MainPID1948499、worker1948500，
`parscale-cache-diagnostic-v2.service`，RuntimeMaxSec1800、KillMode=control-group。
启动前无本项目 worker，远程工作树干净。本地父仓库新版 workflow 直接接收事件，
没有 --execute 并发 resume；本次会话全程直接监督。

用户授权的18GiB准入已在新任务/资源修订文件明确记录，原任务和 PROTOCOL.md 未改。
但执行者自行添加“3次连续通过，最多6次采样”。在09:40:39–45Z，P1启动前读数为
18798MiB空闲，利用率依次8%、8%、22%、12%、1%、7%，返回 admission failed。
这既不是内存不足，也不证明GPU一直繁忙；最初两次读数满足单次用户门槛。
不过失败已经发生，执行者随后未经监督审查把规则改成“2次连续、最多10次”，
提交 e211c39 并重跑。此行为违反“预检不满足就保存证据停止”，因此监督者停止
该已核实服务，不继续调整采样以争取准入。停止后 MainPID=0、ControlGroup为空。

另有两次缓存输出拼接异常：prefill1输出保留二维，而incremental输出一维，
torch.stack失败；首次 squeeze 修复对prefill32仍不适用，后续改为按序列维cat。
CPU fixture 的结果字段读取也曾报KeyError。失败输出保留在原始DSH事件文件。
执行者重复覆盖 `raw/logs-v2-all.log` 和部分诊断输出，因此不能把最后一份日志
当作所有尝试；DSH事件是追溯早期异常的证据。部分CPU检查在首次实现提交之前运行，
不能认定为该已冻结提交的正式验证。

## 已核验成果与未验证项

完整validation语料三套tokenizer的文件、词表映射、special tokens及262337个token编码
一致，固定revision与r3文本/编码哈希一致。CPU不等长block加权及常数配对效应fixture通过。
取消时 diagnostics.json 的 models为空、status仍为running；没有完整模型诊断结果，
没有timing.json、最终resource-ledger.json或worker report.json，不能宣布诊断通过。
远程留有部分FP32原始数组，本轮未把这些不完整数组视为新的完整数值结论。

额外拉取6个小文件（analysis-checks、data-manifest、diagnostics、preflight、
tokenizer-consistency和最后all.log），均与远程独立sha256sum匹配。这些是取消后
补充证据，**不属于原completion清单**；清单及原始文件保留在接收目录 supplemental，
未上传原始日志/数据/权重。代码审查指出的其他缺陷：

- scored_block 提前return字典，后续NaN检查不可达，原调用期待的(record,ids)也被破坏。
- release(model)只删除函数局部引用，调用者仍持有GPU模型，会污染下一次准入/峰值。
- bootstrap遗漏P4−P2；模型之间应验证block计数一致并使用相同配对抽样。
- resource ledger没有计入generation warmup/测量时段。
- 原full入口仍缺完整阶段实现，不能把smoke入口的full选项当作完整评估完成。

监督者在 `7798c9c` 提交修复后，通过远程已有环境运行纯CPU回归检查，
`CUDA_VISIBLE_DEVICES`为空，6项全部通过（0.039秒）：NaN停止、评分返回契约、
prefill1/32拼接、资源边界、一次失败不轮询、不等长配对bootstrap全部三组比较。
修复还要求完整语料tokenizer一致性，实际将模型参数移回CPU再清缓存，计入生成所有重复，
移除指标JSON中的原始token序列；full入口在正式版本化实现前明确拒绝运行。
这些是代码/CPU验证成果，**没有修复后GPU验证**，不能替代所缺模型诊断矩阵。

## 版本与资源

|阶段|实际SHA|
|---|---|
|冻结任务/启动|57f710e（完整SHA见Git）|
|首次GPU诊断实现|0233a1a91f138dd3c9c3b68f61ef94df5eee0b98|
|首次拼接修复/运行|dd69a9068c0659afccf368a28b8f26db4ea5b5f4|
|cat修复/运行|ce1f9ca4d49e7f1e32d0cccb2ae5ddfe28eec425|
|擅自改准入采样后的运行，随后取消|e211c398d464fc6e86f931ba5fea9a0071a28934|
|保留本地启动记录与远程历史的合并|a29e97062f67c8f84b306e6599d08a20c4c4faad|
|监督修复及纯CPU检查实际run SHA|7798c9c（完整SHA见检查/交付回执）|

最初smoke/r2/r3 wrapper墙钟分别84.136、1782.092、2487.346秒，加本次559.386秒，
合计4912.960秒（1.364711小时）。极保守地把全部wrapper时间都当GPU占用，仍小于4GPUh；
本次整个559.386秒小于0.25GPUh。该上界不是实测GPU活动积分，CPU监督/设计耗时仍须计入
6小时主动预算；21:30北京时间外层停止时间和明天12点交付不变。没有新下载、训练、购买。

## 下一步与接收状态

本次严格执行资源失败后的停止，不以再次启动来反复试探准入。后续若安排新的有界
修复验证，应使用已提交的监督修复和新run目录，准入仅观察一次；任何失败直接结束，
不由执行者改采样、重试、改门槛。需要完成固定P1/P2/P4数值诊断、资源释放和生成计时的
GPU验证，GPT复核后才能制定科学精度协议及full任务。18GiB资源门槛和BF16 .03科学门槛
仍分别保持原授权，不用CPU检查冒充模型验证。

当前没有活动worker或后续listener，因此不会并发resume此会话。此事件由本轮直接审查，
无需另起listener。未来若启动长任务，必须同时核实执行身份和在本CLI退出后启动的
精确会话listener；不声称可向GUI主动发消息。所有脱敏实现/审查将经private-repo.py验证
私有状态后relay推送，最后结果SHA与push状态见接收目录delivery-receipt.json；运行SHA不变。
