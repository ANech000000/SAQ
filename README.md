# SAQ: Shape-Aligned Lookup Table Quantization

Research code for **SAQ: Shape-Aligned Lookup Table Quantization for Efficient LLM Inference**, based on [Any4](https://github.com/facebookresearch/any4). Historical code identifiers include CSD-LUT, Method30, and Post-AWQ 22A.

## Source code

Download and extract **[SAQ-source-20261117.zip](SAQ-source-20261117.zip)**. The archive preserves the source directory structure and contains 171 syntax-checked Python files, CUDA/TinyGemm source, experiment entry points, configurations, documentation, and selected result evidence.

```text
code/any4/                  Core source, experiments, CUDA, and evaluation scripts
docs/                       Reproduction guide, experiment index, and limitations
results/hellaswag_20261116/  Corrected HellaSwag results and provenance
results/historical/         Selected historical summaries
SOURCE_MANIFEST.json        Source file inventory and SHA256 hashes
LICENSE                     Original Any4 license
```

The archive excludes model weights, datasets, calibration caches, environments, compiled binaries, historical debug logs, and the paper manuscript.

## Reproduction

See [REPRODUCTION.md](REPRODUCTION.md) and [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md). Paths in these documents refer to the extracted archive. Run experiments from `code/any4/` in a Linux/CUDA environment after preparing the required models, datasets, and calibration statistics. The historical absolute server paths and dependency list require adaptation; this is not a verified one-command fresh-machine installation.

Core entries inside the archive:

- Shared LUT / structured outliers: `code/any4/experiments/method30_outlier/`.
- Alternating refinement: `code/any4/experiments/method30_postawq_refine/`.
- Shape metric / scaling search: `code/any4/experiments/method24_shape_w2/` and `method25_linear_recon_shape/`.
- Packed inference: `code/any4/experiments/method30_outlier/packed_runtime/`.
- Weight-only baselines: `code/any4/experiments/method30_weightonly_1b/scripts/`.
- Corrected HellaSwag evaluation: `code/any4/server_eval/`.

## Corrected HellaSwag results

Zero-shot raw accuracy using the Table I lm-eval 0.4.5 entry, all 10,042 validation examples, batch size 8, and maximum length 2048:

| Method | Accuracy (%) |
|---|---:|
| FP16 | 47.73 |
| RTN | 45.08 |
| GPTQ | 46.14 |
| AWQ | 46.38 |
| Any4 | 46.37 |
| CSD-LUT / SAQ | **46.86** |

The FP16 result exactly reproduces Table I. See [comparison.csv](comparison.csv), [validation.json](validation.json), and the per-method raw JSON inside the archive. GPTQ's historical quantization was reproduced in its original runtime, then transferred through an in-memory pipe with all 112 packed layers hash-verified before evaluation in the common runtime.

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the historical Any4 cross-table model difference, CSD definition/implementation review, and the scope of system claims. Older result protocols are preserved separately and should not be mixed with this corrected column.

## License and attribution

The original Any4 Attribution-NonCommercial 4.0 International license and existing source notices are retained. See [LICENSE](LICENSE) and [NOTICE](NOTICE). No model or dataset license is granted by this repository.
