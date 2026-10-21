# Parallel Scaling 最小复现结果

2026-10-21 19:07（北京时间）完成 GPU 全量评分，独立 CPU 复算通过。官方约 0.7B 的 P1/P2/P4 在固定 WikiText-2 测试集上呈现损失依次下降的趋势，BF16 与 FP32 结论一致，满足测试前固定的最小验证判据。结果支持这些 checkpoint 在本语料和规模下的趋势；没有重新训练模型，不能据此宣称完整复现论文的 scaling law 或效率倍数。

本次 run 为 `parscale-20261021-full`，共293个连续文本块、每模型/精度298,785个计分 target tokens，六组均使用同一数据和 tokenizer。上下文1024，batch1，保留末尾短块，NLL按token加权（越低越好）。完整协议见 [protocol-v2.md](experiments/parscale-20261021-full/protocol-v2.md)。

| 模型 | 参数量 | BF16 NLL | BF16 PPL | FP32 NLL | FP32 PPL |
|---|---:|---:|---:|---:|---:|
| P1 | 671,948,032 | 2.965453 | 19.4035 | 2.965116 | 19.3969 |
| P2 | 674,330,498 | 2.932469 | 18.7739 | 2.932292 | 18.7706 |
| P4 | 676,712,068 | 2.899778 | 18.1701 | 2.899535 | 18.1657 |

主要比较P2−P1：BF16 NLL差−0.032983 nats/token，相对下降1.1123%，配对文本块bootstrap的95%CI为[−0.036439, −0.029395]。P4−P1为−0.065675，CI[−0.069185, −0.062145]；P4−P2为−0.032691，CI[−0.036057, −0.029397]。2000次重采样，固定seed20261021；所有模型共享抽样块索引，末块按实际token数加权。P2下降≥0.5%、其CI上界<0、且P4≤P2，三项预先判据均满足。

FP32对照：P2−P1相对下降1.1070%，NLL差−0.032824，CI[−0.036319, −0.029227]；P4继续下降，亦满足同一判据。BF16相对FP32的绝对NLL偏差仅0.000178–0.000337 nats/token，主要结论不依赖这两种精度选择。

作者公开24点数据的数值重拟合已复核：Pile R²约0.998676，Stack-v2-Python R²约0.997833。这是公开数据复算，不是新训练数据；三份checkpoint不足以重新拟合/证实完整多参数 scaling law。

本机型能运行这些模型：单张RTX3090 24GB，实际全量评分峰值PyTorch reserved显存4.361GiB、allocated显存3.825GiB，低于16GiB分配器上限。该值不包括其他用户进程和全部CUDA上下文开销。验证预检36.84秒，完整BF16+FP32评分和生成计时398.36秒，二者共435.20秒，保守占卡时间上界0.12089 GPU小时；包装流程、历史诊断和准备耗时另计。

128-token输入、固定生成32token，3次预热后10次重复，中位时间P1/P2/P4分别1.2208/1.2671/1.2965秒，p90分别1.5758/1.2847/1.3163秒。同一GPU上存在其他用户负载，且这里只比较同规模checkpoint，这些计时仅作运行观察，不支持复现论文的22倍/6倍效率结论。

原BF16增量缓存一致性门槛0.03仍未通过：v4的P1/P2/P4最大logit误差为0.203125/0.21875/0.1875；FP32约2.3e−5至2.9e−5。失败记录未修改。本轮使用测试前单独固定的teacher-forced NLL协议，每块重新初始化官方learned-prefix cache，块间不传递cache，并增加全量FP32对照。新入口在六项validation NLL全部重现v4（误差≤1e−5）后才访问test；没有按test结果调参或重跑。

置信区间仅反映本测试语料文本块的抽样变化，邻接块相关性、checkpoint训练seed未知和潜在预训练语料重叠限制推断；它不表示独立训练seed的不确定性。

独立审查使用不导入评估器的CPU脚本，从每块记录复算加权NLL、PPL、三组配对bootstrap及计时统计，并检查模型间块哈希、token数、有限性、manual/label loss一致性及pretest/code身份。该审查通过；它不是第二次独立模型推理。[独立结果](experiments/parscale-20261021-full/supervisor-independent.json)与[审查记录](experiments/parscale-20261021-full/review.md)列明证据范围。

实际GPU运行与独立CPU审查代码commit均为`5acb76e310e15f76e4ee8c50adaedd83e7b8aa7a`（开跑时工作树干净），不能用后续结果/报告commit替代。v4 GPU诊断运行`c6a28df97623a4ad9f40ea5403178301eb085c47`，v4结果`dff8edd24aedfb85e53b804a6b73f88655ad6d32`；更早失败和修复保留在各run的审查记录中。完整测试worker结果commit为`ceb71c1f3fbcb6f0a372378c67f2fa859f029f53`；完成事件及11份交付文件校验通过，服务已正常退出并释放本任务GPU资源。当前报告提交可用`git log -1 --format=%H -- RESULTS.md`定位；最终HEAD、服务退出状态及private验证/push回执保存在本地`runs/parscale-20261021-full-received/delivery-receipt.json`，版本摘要见本轮review.md。

远程证据目录：`/path/to/parscale-reproduction/runs/parscale-20261021-full`。模型与数据缓存：`/path/to/parscale-runtime`。原始日志、数组、模型权重和数据未上传Git；本地接收目录保存完成事件与通过哈希校验的小型交付文件。

复算命令（远程既有环境，禁用GPU）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=/path/to/parscale-runtime/pkgs:/path/to/python/site-packages \
python3 scripts/review_full_results.py --run-dir runs/parscale-20261021-full --out /tmp/parscale-full-review.json
```

完整运行入口为`scripts/run_full_evaluation.sh`，冻结task和locks随仓库保存。现有run包含一次性启动锁，勿重跑已完成目录。
