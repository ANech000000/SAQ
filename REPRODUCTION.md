# 代码阅读与复现入口

所有路径相对于 `code/any4/`。本次只静态核对命令参数，没有执行下列模型或 GPU 实验。

## 先读哪些代码

| 作用 | 路径 |
|---|---|
| 原项目入口与许可证 | README.md、LICENSE、requirements.txt |
| 形状差异、等价缩放 | experiments/method24_shape_w2/method24_shape.py |
| alpha 候选搜索 | experiments/method25_linear_recon_shape/full_search.py、layer_timing.py |
| 输出误差 + CSD 选择 | experiments/llama3_8b_reproduction/select_alpha.py |
| 正式 Post-AWQ 22A 重构 | experiments/method30_outlier/method30_outlier_quant.py、runner.py |
| 合成输入 smoke | experiments/method30_outlier/smoke.py |
| 状态、交替重构和缓存 | experiments/method30_postawq_refine/method30_refine.py、runner.py、state_cache.py、stats_bundle.py |
| 1B Fisher/Hessian 对角统计采集 | experiments/method9/collect_fisher.py |
| 8B 激活及 Fisher 统计采集 | experiments/llama3_8b_reproduction/collect_stats_sources.py |
| 8B 主表 | experiments/llama3_8b_reproduction/runner_8b_postawq_table.py、eval_suite.py |
| 打包布局和加载 | experiments/method30_outlier/packed_runtime/packed_format.py、packed_model.py |
| checkpoint 导出 | 同目录 export_checkpoint.py、export_tinygemm_checkpoint.py |
| 融合 CUDA 内核 | 同目录 psop_tinygemm_ext/setup.py、csrc/ |
| packed 精度与速度 | 同目录 evaluate_packed_model.py、benchmark_model.py、run_service_benchmark.py |
| 缩放消融 | experiments/method30_csd_ablation/ |
| 权重量化比较 | experiments/method30_weightonly_1b/scripts/ |
| 跨架构、异常值、组大小实验 | experiments/paper_additional_20260818_v1/scripts/ |

基本执行顺序是：准备模型和离线数据 → 采集激活/对角统计 → 搜索并选择缩放 → 在缩放域每组分离一个异常值 → 固定缩放与异常值位置后更新 Q、C、A/B → 评估 → 导出和验证 packed 表示。不要把全部变量描述成同时联合优化。

## 环境与输入

历史环境以 Linux + NVIDIA CUDA GPU 为基础。需要 PyTorch、Transformers、NumPy、scikit-learn、joblib、threadpoolctl、safetensors、Triton 及评测依赖；服务测试还用到 FastAPI、uvicorn、requests。首先参考原 requirements 和对应脚本导入，再为目标机器建立环境。原 requirements 不是完整环境锁文件，不能据此保证版本完全复现。

CUDA 扩展需在目标 GPU 环境重新构建，例如从 `experiments/method30_outlier/packed_runtime/psop_tinygemm_ext/` 执行其 setup.py；TinyGemm 自身位于根目录 `tinygemm_lib/`。包中保留源码，不带本机二进制。

需要另外准备：

- 原始 Llama/Qwen/Phi 等模型及 tokenizer，按各模型许可取得。
- WikiText-2、C4、PTB、CodeParrot 与相关下游评测数据。
- 激活缓存、Fisher/输入 Hessian 对角统计、stats bundle、最终量化状态；可由归档脚本产生。
- 若验证 packed 推理，需要另外导出或取得部署 checkpoint；8B 历史路径为 `/data/lsz/any4/experiments/method30_outlier/checkpoints/llama3_8b_postawq_22a/tinygemm_fp16_meta`。

多数脚本默认离线加载，并保留 `/data/...`、`/home/ubuntu/...` 路径。CLI 已提供的路径请通过参数传入；没有参数的常量需在自己的工作副本中适配。归档文件本身保持不变。

## 已核对参数的命令示例

在 `code/any4/` 工作目录执行。以下 `MODEL`、`ACT`、`FISHER`、`ALPHA`、`BUNDLE`、`STATE`、`OUT`、`CKPT` 都是说明变量，须先设置为真实路径。为输出指定新的目录，避免覆盖归档结果。单 GPU 编号应按目标机器调整。

合成输入检查，不需要原模型权重（仍需对应计算依赖）：

```bash
python experiments/method30_outlier/smoke.py
```

采集 1B 对角统计及运行候选搜索（激活 ACT 须先由 method24_shape_w2/runner.py 等入口准备）：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/method9/collect_fisher.py \
  --model-path "$MODEL" --output "$FISHER" \
  --nsamples 128 --seqlen 2048 --seed 0

CUDA_VISIBLE_DEVICES=0 python experiments/method25_linear_recon_shape/full_search.py \
  --model-path "$MODEL" --layers 16 --physical-gpu 0 \
  --activation-cache "$ACT" --output-dir "$OUT/search" \
  --group-size 128 --parallel-jobs 16

python experiments/llama3_8b_reproduction/select_alpha.py \
  --search-results "$OUT/search/full_search_results.json" \
  --regularization 0.3 --output "$ALPHA"
```

虽然选择脚本位于 8B 目录，它可读取相应 1B/8B 搜索记录。它的输出键/描述固定为 method27_lambda_0p3，因此上述示例固定 lambda=0.3。1B 为 16 个 block；8B 为 32 个。历史 alpha 选择与候选 JSON 已附在 `logs/llama3.2-1b/`；其中 unit_file 仍为原服务器绝对路径，迁移时需映射或重新搜索。

直接复现原始 Post-AWQ 22A 路径：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/method30_outlier/runner.py \
  --variant exact --outlier-selection-domain post_awq \
  --physical-gpu 0 --model-path "$MODEL" \
  --alpha-selection-path "$ALPHA" --activation-cache "$ACT" \
  --fisher-path "$FISHER" --max-iter 10 --output-dir "$OUT/direct"
```

必须明确 `--outlier-selection-domain post_awq --variant exact`；脚本默认是 pre_awq/all，不能依赖默认值复现此版本。数据缓存、group size 等参数见该入口 argparse 及 config.json。

需要可导出的最终状态时，使用 stats bundle 与 refine：

```bash
python experiments/method30_postawq_refine/stats_bundle.py \
  --model-path "$MODEL" --alpha-selection-path "$ALPHA" \
  --alpha-key method27_lambda_0p3 \
  --activation-cache "$ACT" --fisher-path "$FISHER" --output "$BUNDLE"

CUDA_VISIBLE_DEVICES=0 python experiments/method30_postawq_refine/runner.py \
  --phase all --physical-gpu 0 --model-path "$MODEL" \
  --stats-bundle "$BUNDLE" --state-cache-dir "$STATE" \
  --initializer sklearn --checkpoints 10 --target-iters 10 \
  --output-dir "$OUT/refine"
```

显式固定 sklearn 初始化和 10 轮；refine 默认包含其他轮数。上述 bundle 示例针对 1B；8B 的动态 224-Linear bundle 入口为 `llama3_8b_reproduction/build_stats_bundle.py`，不要直接套用 1B 层数约束。

已取得正确的 8B packed checkpoint 后：

```bash
CUDA_VISIBLE_DEVICES=0 PSOP_TINYGEMM_MAX_M=64 \
python experiments/method30_outlier/packed_runtime/evaluate_packed_model.py \
  --checkpoint-dir "$CKPT" --physical-gpu 0 \
  --datasets wiki2,c4,ptb --seqlen 2048 --c4-samples 256 --seed 0 \
  --output "$OUT/packed_ppl.json"
```

## 已知迁移限制

- 当前 packed 模型/导出路径含 8B 的 224 个 Linear 约束，不能据此宣称任意架构均可直接部署。
- packed C4/PTB 评测有源码常量路径，没有对应 CLI 路径参数。
- refine 使用 Linux CPU affinity/NUMA，Windows 适合阅读和整理，不是本次已验证的复现实验环境。
- 权重量化对比的 official_gptq_runner.py 引用 `method30_weightonly_1b/third_party/auto_gptq`。该第三方目录未收录；需要另行安装/准备对应依赖并适配入口。
- 不能直接运行会固定写 `runs/smoke/` 等目录的旧 smoke/launch 脚本来覆盖证据；先在新工作副本中核对输出路径。
- 本次只做归档和静态检查，没有验证新环境安装、数值精度、CUDA 编译或性能。请把这些作为后续独立复现步骤。
