# 论文实验与文件索引

下表中的路径均相对于 `code/any4/`。原始 JSON、CSV、日志按服务器目录保留；`evidence/local_snapshot/` 是本机已有下载结果的额外备份，不替代服务器快照。历史候选、失败试验及机制诊断均不自动成为论文最终证据。

| 论文内容 | 结果位置 | 使用说明 |
|---|---|---|
| Table I：1B 数值格式主表 | experiments/method30_postawq_refine/runs/any4_table_1b/；同目录下 runs/table1_official_reproduction/ | full、baselines 等目录保留逐任务结果 |
| Table I：8B 数值格式主表 | experiments/llama3_8b_reproduction/runs/table1_postawq_22a/results.json；runs/ppl/ | 标准反量化 PPL 见 runs/ppl/postawq_22a.json |
| Table II：1B weight-only 比较 | experiments/method30_weightonly_1b/runs/comparison_20260816_final/comparison.csv、comparison.json、manifest.json | 最终汇总；原始来源见同级 comparison_20260816_v3、official_gptq_v4 |
| Table III：Qwen/Phi 泛化 | experiments/paper_additional_20260818_v1/cross_arch/qwen2_5_1_5b/；cross_arch/phi2/ | fp16、rtn、csd_lut_output_csd_v1 等子目录；Gemma 不在当前正文主表 |
| Table IV：重构组件 | experiments/method22/；experiments/method30/；主表目录中的 FP16/Any4 | 论文累计值为 9.750678、10.618476、10.463196、10.408120，需按运行设定归因 |
| Table V：缩放选择 | experiments/method30_csd_ablation/runs/csd_selection_ablation_20260807/paper/paper_table.csv、paper_table.json | 1b/8b 下有四策略逐模型结果 |
| Table VI：异常值 | experiments/paper_additional_20260818_v1/outlier/no_exact_c4/；正式 exact 主表结果 | 不把重构、缩放和异常值变化混为一个单变量收益 |
| Table VII：组大小 | experiments/paper_additional_20260818_v1/group_size/；packed_bpw/ 中的文本统计 | 不含统计所依赖的权重/最终状态文件 |
| 早期原始 1B Post-AWQ 22A | experiments/method30_outlier/runs/wiki2_ppl/post_awq/method30_post_awq_22a_exact.json、results.json | 早期 10.2880029678，与统一主表 10.2877893448 分开保存 |
| 8B 存储与系统指标 | experiments/method30_outlier/runs/llama3_8b/postawq_22a_packed/paper_metrics_v2.json | v2 包含更完整实际文件体积口径；同目录含 paper_kernel/model/service.csv |
| 最终 hybrid m64 系统实验 | 上述目录的 hybrid_final_m64_20260805/summary.json、paper_accuracy.csv、paper_model_median.csv、paper_service_median.csv | 3 次同 GPU 配对重复的中位数汇总，有源文件/provenance 记录 |
| 历史机制诊断 | experiments/method30_csd_mechanism/ | 保留追溯，不恢复为当前论文核心 claim |
| 1B alpha 选择依据 | logs/llama3.2-1b/wiki2_method27_lambda_sweep_20260714/batch2_gpu2/lambda_selections.json | 原配置指定 lambda=0.3；候选记录在 wiki2_methods25_27_20260714_165642/search/ |

## 已核对的关键数值

1B weight-only 最终比较表中，SAQ/CSD-LUT 的 WikiText-2、C4、PTB 分别为 10.2877893448、12.4038791656、18.5910797119。

8B 标准反量化结果中，WikiText-2、C4、PTB 分别为 6.3587446213、8.1936483383、11.4735860825。

8B hybrid m64 对应结果分别为 6.3588767052、8.1932687759、11.4735860825；相对标准反量化的最大绝对差约 0.0003795624。

同一最终系统汇总记录：加载后 FP16 allocated memory 为 15316.5083 MiB，packed 为 5760.6333 MiB，降低 62.3894%；batch-1 三种 prompt 长度的 decode speedup 约 1.4772、1.5118、1.5483。prefill 不能据此宣称加速。

数值来源是历史结果文件。本次整理没有重新测量，未扩大历史结论。
