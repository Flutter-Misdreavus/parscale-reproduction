# parscale-20261021-full：监督审查

决定：**接受本次最小checkpoint比较结果；不新增GPU实验。** 用户要求的期限内最小验证已完成。BF16与FP32均满足测试前固定的判据，结论限定于三个官方checkpoint和本测试语料。公开数据重拟合亦已完成；不声称独立训练或完整scaling law/效率倍数复现。

完成事件：`b990b64e22d2139e4aafff1c1de838ae3883bde5103420dfccacc54c5bbf687a`；状态SUCCEEDED，完成UTC `2026-10-21T11:08:55.307705+00:00`。任务身份与本地冻结task一致，11份交付文件的字节数/SHA256复核通过；完整记录见evidence.json。原始DSH日志和模型/数据没有进入Git。

CPU独立审查在CUDA不可见环境运行已提交的`scripts/review_full_results.py`，实际代码版本为`5acb76e310e15f76e4ee8c50adaedd83e7b8aa7a`。该脚本不导入worker评估器，从293块记录复算六组token-weighted NLL/PPL、全部配对bootstrap和生成计时统计。验证pretest身份、test块数/哈希、每块目标数、有限性、manual-label loss误差以及实际run SHA，全部通过。每模型/精度计分298785个target。

独立审查输入full-results SHA256为`95780208b0b47e33a000767985434bd4a0a8507e1cd1128f1a9d5950983e9282`，与完成事件清单及收到的文件相同。独立输出supervisor-independent.json同时与worker脱敏metrics逐项交叉核对。审查复算统计，不重跑模型推理；v4原始数组复算提供先前数值诊断证据。

BF16主要比较P2−P1：−0.0329833324 nats/token，95%CI [−0.0364394946, −0.0293947261]，相对下降1.1123%；P4≤P2。FP32：−0.0328242537，CI [−0.0363190659, −0.0292268434]，下降1.1070%，P4≤P2。两种精度都满足≥0.5%下降、CI上界<0及P4≤P2三项判据。没有根据测试结果调整评价或重跑。

原BF16增量缓存0.03门槛仍失败；单独固定的protocol-v2在六项validation NLL复现（最大误差1.1933e−7，阈值1e−5）后，评估每块重新初始化官方前缀缓存的teacher-forced路径，并做FP32全量敏感性对照。失败记录保留，不能把此次成功写成原门槛通过。checkpoint训练seed未知、语料可能重叠、邻接文本块相关性限制CI解释。共享GPU生成计时仅为描述性结果。

运行代码/task commit：`5acb76e310e15f76e4ee8c50adaedd83e7b8aa7a`；worker结果commit：`ceb71c1f3fbcb6f0a372378c67f2fa859f029f53`，该结果提交仅含environment.json、metrics.json、results.md。当前审查前HEAD同worker结果commit；审查后HEAD与实际private核验/push状态记录于本地`runs/parscale-20261021-full-received/delivery-receipt.json`。报告提交可通过`git log -1 --format=%H -- experiments/parscale-20261021-full/review.md`定位。worker报告的pending是生成报告时的relay状态，不能当作最终推送状态。

预检36.842秒，完整评估398.356秒，保守占卡上界0.12089GPUh；完整wrapper587.770秒。已登记七次wrapper累计6613.323秒（约1.8370小时，包括下载等待/DSH报告），其内GPU工作低于4GPUh预算；没有完整主动时间积分，不以wrapper时间冒充全项目6h计量。全部GPU任务已结束且在今日21:30外层停止时间之前。远程runs9.2GiB、runtime8.6GiB，总量低于25GiB磁盘额度。

最终服务`parscale-full.service`：inactive、MainPID=0、Result=success、ExecMainStatus=0、ControlGroup空。当前对话通过MCP桥实际收到完成事件并完成审查，无后台Codex审查器，无下一轮自动实验。宿主原生har-remote工具在本轮仍未暴露；接收使用stdio MCP客户端。此证据证明本轮对话内闭环，不保证已结束会话的主动唤醒。
