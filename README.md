
# Gated GRIT

**Curvature-guided rank-wise gating for parameter-efficient adaptation of multimodal vision-language models.**

Gated GRIT is an experimental extension of **GRIT (Geometric Reprojection Instruction Tuning)** for multimodal large language models. It combines K-FAC-based curvature-aware optimization with a rank-wise gating mechanism, called **SuperGate**, to modulate how low-rank adaptation directions contribute to optimization.

> **Project status:** Research prototype. Experiments are ongoing and the Gated GRIT manuscript is in preparation. The preliminary numbers below are small-split smoke-test results, not final benchmark results.

---

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [SuperGate Mechanism](#supergate-mechanism)
- [Models and Tasks](#models-and-tasks)
- [Baselines](#baselines)
- [Preliminary Results](#preliminary-results)
- [Experimental Protocol](#experimental-protocol)
- [Repository Organization](#repository-organization)
- [Setup](#setup)
- [Running an Experiment](#running-an-experiment)
- [Evaluation](#evaluation)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations and Current Status](#limitations-and-current-status)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Motivation

LoRA makes fine-tuning large models practical by expressing weight updates through low-rank matrices. However, standard LoRA does not explicitly use the local curvature of the optimization landscape to guide those updates. GRIT addresses this limitation by incorporating K-FAC-based curvature information into low-rank optimization.

Gated GRIT explores a further question:

**Can curvature information also indicate which low-rank directions should receive more or less optimization emphasis?**

The central idea is to use curvature-derived information to construct rank-wise gates and apply them to the K-FAC-preconditioned LoRA gradients. This adds a mechanism for modulating the use of the available low-rank adaptation capacity without introducing a separate set of trainable gate parameters.

The project studies this approach in multimodal settings, where adaptation involves both visual and language components of vision-language models.

## Method Overview

Gated GRIT builds on the GRIT optimization pipeline. At a high level, the implementation follows these steps:

1. **Low-rank adaptation:** LoRA modules are attached to selected projection layers in the vision and language components of the model.
2. **Curvature-statistic collection:** During training, the implementation collects activation- and gradient-derived statistics associated with the LoRA factors.
3. **K-FAC approximation:** These statistics are used to construct and periodically invert small curvature factors. The resulting preconditioners are used to transform LoRA gradients.
4. **SuperGate construction:** Eigenvalue information from the rank-space covariance statistics is normalized to obtain a rank-wise gate vector.
5. **Rank-wise gradient modulation:** The gates scale the corresponding rank dimensions of the preconditioned LoRA-A and LoRA-B gradients.
6. **GRIT optimization:** The optimizer continues with the curvature-aware update and the configured GRIT reprojection and rank-adaptation mechanisms.

The implementation is designed to retain the parameter efficiency of LoRA while using curvature information to guide optimization in the low-rank subspace.

## SuperGate Mechanism

For a LoRA update, the weight matrix is expressed as:

`ΔW = BA`

The implementation maintains curvature-related covariance estimates for the low-rank factors. The SuperGate mechanism uses eigenvalue information from the rank-space covariance associated with the LoRA-A side.

Let the non-negative eigenvalues be λ₁, λ₂, ..., λᵣ. The normalized rank-wise gate values are computed as:

`gᵢ = λᵢ / Σⱼ λⱼ`

A configurable lower-floor operation is subsequently applied, followed by renormalization. The resulting gate vector `g` contains one value for each LoRA rank direction.

After K-FAC preconditioning, the implementation applies the same rank-wise weights to the corresponding dimensions of both LoRA factors:

```text
∇Ã[i, :] = g[i] × ∇Ã[i, :]
∇B̃[:, i] = g[i] × ∇B̃[:, i]
```

Here, ∇Ã and ∇B̃ denote the preconditioned gradients.

In the current implementation, the gate vector is derived from curvature statistics rather than being introduced as an independently learned trainable parameter.

The gates are associated with individual LoRA modules and are refreshed when the relevant curvature factors are updated. This allows module-specific rank-wise modulation during optimization.

**Important distinction:** SuperGate is a gradient-modulation mechanism, not hard pruning of LoRA ranks. Effective-rank and sparsity analyses are separate diagnostics of the resulting gate distributions.

## Models and Tasks

The experimental notebooks cover the following model and task combinations.

### Vision-Language Models

- [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it)
- [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

Access to gated model repositories may require accepting the model terms on Hugging Face.

### Tasks and Datasets

| Task | Dataset | Purpose |
|---|---|---|
| Image Captioning | [bitmind/MS-COCO](https://huggingface.co/datasets/bitmind/MS-COCO) | Generate natural-language descriptions of image content |
| Visual Question Answering | [VQAv2](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) | Answer questions grounded in image content |
| Referring-Expression / Region Description | [RefCOCO](https://huggingface.co/datasets/lmms-lab-encoder/RefCOCO) | Describe the image region specified by a bounding box |

The exact split handling and preprocessing are defined in the corresponding experiment notebook. For RefCOCO, the implementation uses the dataset's region annotations and referring expressions; the prompt asks the model to produce a short description of the selected region.

## Baselines

The experimental comparison includes:

- **LoRA:** Standard low-rank adaptation.
- **AdaLoRA:** Adaptive allocation of low-rank capacity.
- **GRIT:** Curvature-aware low-rank optimization using K-FAC-based preconditioning and Fisher-guided reprojection.
- **Gated GRIT:** GRIT with the additional SuperGate rank-wise gradient-modulation mechanism.

For a meaningful comparison, methods should use matched model checkpoints, dataset subsets, preprocessing, random seeds, target modules, training budgets, generation settings, and evaluation code. Any deviation should be reported alongside the results.

## Preliminary Results

The following result is from an early **Gemma 3 4B IT image-captioning smoke test** on `bitmind/MS-COCO`.

### Gated GRIT — Preliminary Smoke Test

| Setting | Value |
|---|---:|
| Model | Gemma 3 4B IT |
| Training examples after filtering | 32 |
| Validation examples after filtering | 4 |
| Test examples after filtering | 4 |
| LoRA rank | 4 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Training steps | 24 |
| Learning rate | `2 × 10⁻⁶` |
| Precision | BF16 |
| K-FAC update frequency | 10 steps |
| Configured reprojection frequency | 300 steps |

### Test Metrics

| Metric | Preliminary Value |
|---|---:|
| Test loss | 25.21310 |
| ROUGE-L | 0.3376 |
| METEOR | 0.3212 |
| BERTScore F1 | 0.6669 |

**How to interpret these numbers:**

- The test set contained only four examples. These values are highly sensitive to individual examples and should not be treated as reliable estimates of generalization.
- The run completed 24 optimizer steps. It therefore did not reach the configured 300-step reprojection interval. It is an implementation smoke test, not evidence of the full reprojection schedule's effect.
- The result is not presented as evidence that Gated GRIT outperforms LoRA, AdaLoRA, or GRIT. A controlled comparison requires matched full-scale runs.
- These numbers are retained here to document an early working experiment. They will be replaced or supplemented by results from the larger experimental protocol once those runs and analyses are complete.

No claim of benchmark superiority is made from this preliminary run.

## Experimental Protocol

The larger experimental protocol being used for the main evaluation is:

| Component | Planned Setting |
|---|---|
| Training split | 8,000 examples |
| Validation split | 1,000 examples |
| Test split | 1,000 examples |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Training epochs | 3 |
| Precision | BF16 |
| Validation during final training | Disabled |
| Best-checkpoint selection | Not used |

With 8,000 training examples, an effective batch size of 4, and three epochs, the nominal training budget is:

**Total optimizer steps = (8,000 / 4) × 3 = 6,000 steps**

This assumes the full training split is retained and the dataloader does not drop additional examples.

The validation split is retained for evaluation or analysis where applicable, but the final training runs do not perform validation-based checkpoint selection. Final test evaluation is conducted after training using the same generation and metric procedures across methods.

The precise hyperparameters for each method and task are recorded in the experiment notebooks. They should be treated as the source of truth for reproducing a particular run.

## Repository Organization

The experiments are notebook-based. Notebooks are named by method, model, and task where possible. A suggested organization for a cleaned public release is:

```text
Gated-GRIT/
├── README.md
├── notebooks/
│   ├── image_captioning/
│   ├── vqa/
│   └── refcoco/
├── results/
│   ├── preliminary/
│   └── final/
├── figures/
└── requirements.txt
```

The directory layout above is a suggested organization; adapt it to the files actually included in the repository. Do not commit model checkpoints, private access tokens, cached datasets, or large temporary training artifacts unless there is a specific reason to distribute them.

## Setup

A CUDA-capable environment is recommended. The notebooks use PyTorch, Hugging Face Transformers and PEFT, along with metric and dataset utilities.

A starting installation command is:

```bash
pip install torch torchvision
pip install transformers peft accelerate datasets
pip install numpy scikit-learn pillow requests
pip install rouge-score bert-score nltk pyemd
pip install git+https://github.com/AIPHES/emnlp19-moverscore.git
```

The notebooks may install or patch additional dependencies for compatibility with the runtime. Exact package versions should be pinned in a release-specific `requirements.txt` after the environment used for the final experiments has been frozen and tested.

### Hugging Face Authentication

Some model and dataset repositories require authentication or acceptance of access terms.

1. Accept the relevant model/dataset terms on Hugging Face.
2. Authenticate in the environment using the Hugging Face CLI:

```bash
hf auth login
```

3. Verify that authentication works:

```bash
hf auth whoami
```

Do not hard-code a Hugging Face token in a notebook or commit it to GitHub. Use the environment's cached credentials or a secret-management mechanism.

## Running an Experiment

1. Clone the repository and install the dependencies.
2. Configure Hugging Face access for the model and dataset being used.
3. Open the notebook corresponding to the desired method, model, and task.
4. Review the configuration cell, especially:
   - Model identifier
   - Dataset and split selection
   - Random seed
   - LoRA rank, alpha, dropout, and target modules
   - Batch size and gradient accumulation
   - Learning rate and number of epochs
   - K-FAC update and inversion settings
   - Reprojection frequency and warmup
   - Gate floor and rank-adaptation settings
   - Generation parameters and evaluation metrics
5. Run the notebook cells in order.
6. Preserve the generated metrics, configuration, logs, and gate-analysis artifacts with the run identifier.

For large experiments, use a persistent terminal session manager such as `tmux` so the training process can continue if the SSH connection closes.

## Evaluation

The notebooks include text-generation metrics such as:

- **ROUGE-L** — lexical overlap based on the longest common subsequence.
- **METEOR** — a metric incorporating token alignment and linguistic matching.
- **BERTScore F1** — contextual semantic similarity.
- **MoverScore** — embedding-based semantic similarity.
- **Task-specific accuracy** — used where the task and annotation format support it.

Metric availability varies by task and notebook. Values should only be compared when the same metric implementation, reference construction, text normalization, and generation settings are used.

For VQA, report the exact accuracy implementation used by the notebook. For region-description tasks such as RefCOCO, simple normalized exact-match accuracy is only a coarse diagnostic because multiple valid referring expressions may describe the same region.

## Reproducibility Notes

For controlled comparisons:

- Use the same train/validation/test subsets across methods.
- Keep the random seed and data ordering consistent.
- Match the effective batch size and training budget.
- Keep model versions, precision, target modules, and preprocessing fixed where possible.
- Use identical decoding settings for final generation.
- Record all method-specific hyperparameters, including K-FAC, reprojection, and SuperGate settings.
- Save the final test predictions as well as aggregate metrics.
- Report failed or incomplete runs rather than silently excluding them.

Exact bitwise reproducibility is not guaranteed across GPU architectures, CUDA/PyTorch versions, or nondeterministic kernels.

## Limitations and Current Status

Gated GRIT is an ongoing research project. The current repository contains experimental implementations and preliminary results. The small-split numbers above are intended to show that the pipeline has been exercised, not to establish a final performance ranking.

The main evaluation still needs to establish:

- Whether the SuperGate mechanism provides consistent benefits under matched full-scale comparisons.
- How gate distributions differ across vision and language components.
- How gate behaviour relates to effective rank and curvature information.
- How results vary across tasks, model families, and random seeds.
- The optimization and runtime overhead introduced by curvature estimation and gating.

The manuscript and full experimental analysis are in preparation. Final claims should be based on completed, controlled experiments rather than the preliminary smoke tests.

## Citation

Gated GRIT is based on the GRIT line of work. Please also cite the relevant dataset and model papers when using this code.

**GRIT reference:**

```bibtex
@misc{saha2026gritgeometryawarepeft,
      title={GRIT -- Geometry-Aware PEFT with K-FAC Preconditioning, Fisher-Guided Reprojection, and Dynamic Rank Adaptation},
      author={Pritish Saha and Chandrav Rajbangshi and Rudra Goyal and Mohit Goyal and Anurag Deo and Biswajit Roy and Ningthoujam Dhanachandra Singh and Raxit Goswami and Amitava Das},
      year={2026},
      eprint={2601.00231},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.00231}
}
```

A citation for the Gated GRIT manuscript will be added when the manuscript is publicly available.

## Acknowledgements

This work builds on the GRIT research direction and explores its extension to multimodal vision-language models. We acknowledge the authors and maintainers of the underlying models, datasets, libraries, and evaluation tools used in the experiments.
