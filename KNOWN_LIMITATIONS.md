# Known limitations and result provenance

## Historical source layout

The source preserves the original experiment directory structure and absolute server paths. It is not a portable one-command package. Models, datasets, activation/Fisher statistics, cached states, and compiled CUDA extensions are not included. The original GPTQ runner expects an AutoGPTQ 0.7.1 source/dependency directory; prepare it separately rather than copying an incompatible binary environment.

Source provenance: the project archive collected on October 28, 2026, plus the November 16 HellaSwag reevaluation scripts. Historical backups, run directories, model files, environments, and logs were excluded from the source upload. Original source and license notices were retained.

## Definition and implementation

The archived paper review identified a CSD normalization mismatch between some manuscripts (mean/std) and the formal candidate-search implementation (min-max). The code uses the latter; this release does not silently change either implementation or manuscript. Scaling ablations attribute the main gain to output-aware scaling, with CSD as a further structural criterion.

## HellaSwag correction

The older weight-only table used a separate token-length-normalized evaluator. The corrected column uses the Table I lm-eval 0.4.5 task and raw accuracy. Both raw accuracy and lm-eval's secondary normalized accuracy remain available in the reevaluation evidence.

Final raw records: `results/hellaswag_20261116/{fp16,rtn,awq,any4,csd_lut}/results.json` and `results/hellaswag_20261116/gptq_bridge_v7/results.json`. Failed compatibility experiments are excluded from these release results.

Five methods exactly reproduce their recorded WikiText-2 PPL in the common runtime. GPTQ's historical producer exactly reproduces PPL 10.714824676513672; its 112 packed layers pass tensor-hash transfer verification. PPL in the newer common inference runtime is 10.714512825012207 (difference -0.0003118515); this diagnostic is kept distinct from the historical table value.

The Table I Any4 run has WikiText-2 PPL 10.6041440964 and raw HellaSwag accuracy 46.3851822346%. The Table II Any4 run has PPL 10.6184759140 and, under the unified reevaluation, accuracy 46.3652658833%. The remaining two-example difference cannot be resolved by changing scoring labels alone; it requires using the same quantization result across tables.

## Other reporting boundaries

- 4.4911 BPW refers to the 8B quantized Linear files; complete checkpoint size and 1B group-size statistics use different accounting.
- The 62.39% memory reduction concerns memory after model loading. Decode improvements do not imply faster prefill.
- Main cross-architecture results cover Qwen2.5 and Phi-2; historical Gemma checks are not promoted to a confirmed main-table result.
- The release is syntax-checked and contains verified HellaSwag result provenance. It does not claim a new full-suite rerun or fresh-environment reproducibility.
