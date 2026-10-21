# 结果复核程序

评分任务在 `34d4a1f9464562d563320c871e1ca1db6f2282d3` 启动；本文件及分析程序是后续独立分析提交，不改变正在执行的评分代码。

`scripts/review_downstream_extension.py` 要求十二组完整结束，逐文件核对 SHA-256，核对模型间逐题与候选身份、提示内容哈希和标签一致，逐项从候选分数重算原始与长度归一化准确率，最后核对执行者汇总值。原始评分文件和答案标签保留在忽略的运行目录中；公开复算文件只包含逐题正确与否。

最终审查还将逐题标签重新组成完整向量，与测试前 `data-manifest.json` 的标签哈希直接比对，并要求候选键集合完全一致、无缺失或额外项。原始准备清单的文件哈希必须与冻结任务一致；放在接收目录中，或用 `--data-manifest` 指定路径。这些是新增的证据检查，不改变评分方法、主要指标或统计判据。

独立的只读数据检查直接用 PyArrow 读取原始 Parquet：ARC 将 `answerKey` 映射到该题 `choices.label` 中的位置，HellaSwag 将 `label` 转为整数，再按原始行号重建标签向量。三项数据的完整标签哈希均与测试前清单一致；该检查没有导入评分辅助模块，也未改写数据。[检查记录](raw-label-audit.json)只保留行数与哈希。

`scripts/bootstrap_downstream_extension.py` 实现已冻结的新三任务分析：ARC-Easy/ARC-Challenge/HellaSwag 权重分别 1/4、1/4、1/2；P4−P1 至少 1 个百分点且配对 bootstrap 95% 区间下界大于零才满足正向判据。重采样 2,000 次，种子 20261022，各任务独立抽题，模型共享抽样索引。另提供排除 ARC-Easy 两道已知训练集重叠题的敏感性分析，不能用它替换官方完整划分的主要分析。

全七任务重建论文六任务组 General：四项旧任务及 HellaSwag 各占 1/6，ARC 两项各占 1/12。这项扩展描述性分析是在旧四任务结果已知后制定，不替代旧主要检验，也不提供训练随机种子层面的不确定性。

准备阶段验证：Python 语法通过；带不同任务大小、非均等权重及正负模型差异的合成例子，其分块 bootstrap 与独立生成完整抽样矩阵的计算完全一致；相同模型的区间为零；ARC 分组权重检查通过。正式结果尚待评分结束及原始产物接收核验。

复核与公开统计复算命令：

```bash
python3 scripts/review_downstream_extension.py \
  --run-dir runs/parscale-20261021-downstream-rest-received \
  --core-outcomes experiments/parscale-20261021-downstream-core/paired-outcomes.json \
  --output experiments/parscale-20261021-downstream-rest/supervisor-independent.json
python3 scripts/bootstrap_downstream_extension.py \
  experiments/parscale-20261021-downstream-rest/paired-outcomes.json \
  experiments/parscale-20261021-downstream-core/paired-outcomes.json \
  --check-against experiments/parscale-20261021-downstream-rest/supervisor-independent.json
```
