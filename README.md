# Gated GRIT

**Geometry-aware parameter-efficient fine-tuning with curvature-derived rank allocation for multimodal vision-language models.**

Gated GRIT is a curvature-aware extension of **GRIT (Geometry-Aware Parameter-Efficient Fine-Tuning)** for multimodal vision-language models. It uses Fisher-spectrum information obtained through Kronecker-Factored Approximate Curvature (K-FAC) to derive rank-wise importance weights and modulate low-rank adaptation.

Unlike conventional gating mechanisms that introduce additional trainable parameters, Gated GRIT derives its gates from curvature statistics already computed by GRIT. The resulting mechanism, referred to as **SuperGate**, assigns different levels of emphasis to individual low-rank directions while retaining the parameter budget of the underlying LoRA configuration.

> **Project status:** Research project. The paper reports experiments on visual question answering using Gemma-3-4B and Llama-3.2-11B-Vision. Further multimodal experiments and analysis are ongoing. The results below are the results reported in the current paper; they should be distinguished from any subsequent or expanded experimental runs.

---

## Table of Contents

- [Motivation](#motivation)
- [Background: LoRA and GRIT](#background-lora-and-grit)
- [Gated GRIT: Method](#gated-grit-method)
- [1. Fisher-Based Curvature Estimation](#1-fisher-based-curvature-estimation)
- [2. Curvature-Derived Gate Computation](#2-curvature-derived-gate-computation)
- [3. Gated Low-Rank Updates](#3-gated-low-rank-updates)
- [4. Effective-Rank Analysis](#4-effective-rank-analysis)
- [Models and Datasets](#models-and-datasets)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Gate and Effective-Rank Analysis](#gate-and-effective-rank-analysis)
- [Repository Organization](#repository-organization)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations and Future Directions](#limitations-and-future-directions)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Low-Rank Adaptation (LoRA) provides an efficient way to fine-tune large pretrained models by constraining parameter updates to a low-rank subspace. However, standard LoRA treats the available rank directions uniformly and relies on first-order gradient optimization without explicitly accounting for the local curvature of the loss landscape.

GRIT extends LoRA by incorporating curvature information through K-FAC-based Fisher preconditioning. This allows optimization to account for the local geometry of the loss landscape.

However, curvature-aware preconditioning alone does not explicitly allocate different levels of importance to individual rank directions.

Gated GRIT investigates the following question:

**Can the Fisher spectrum already estimated during curvature-aware optimization be used to allocate adaptation capacity selectively across low-rank directions?**

The proposed SuperGate mechanism derives rank-wise importance weights from the Fisher eigenspectrum and uses them to modulate the low-rank update. Directions associated with larger eigenvalues receive greater emphasis, while weaker directions are down-weighted without being completely removed.

The mechanism introduces no additional trainable parameters and operates using curvature information already available within GRIT.

## Background: LoRA and GRIT

### LoRA formulation

Given a pretrained weight matrix:

```math
W_0 \in \mathbb{R}^{d \times k}
```

LoRA expresses the adapted weight matrix as:

```math
W = W_0 + \Delta W
```

where the low-rank update is parameterized by:

```math
\Delta W = BA
```

with:

```math
A \in \mathbb{R}^{r \times k},
\qquad
B \in \mathbb{R}^{d \times r}
```

Here, `r` is the adaptation rank, with:

```math
r \ll \min(d,k)
```

Only the low-rank matrices `A` and `B` are trained, while the pretrained backbone remains frozen.

### GRIT and Fisher preconditioning

GRIT incorporates local curvature information into low-rank optimization through a Kronecker-Factored Approximate Curvature (K-FAC) approximation of the Fisher Information Matrix.

Let:

- `∇A` denote the gradient with respect to the LoRA-A matrix.
- `∇B` denote the gradient with respect to the LoRA-B matrix.
- `A_f` and `G_f` denote the Kronecker factors of the Fisher approximation.

GRIT computes the Fisher-preconditioned gradients as:

```math
\widetilde{\nabla A} = A_f^{-1}\nabla A
```

```math
\widetilde{\nabla B} = \nabla B G_f^{-1}
```

These preconditioned gradients incorporate curvature information into the optimization process.

Gated GRIT builds on this formulation by using the Fisher spectrum to determine how individual low-rank directions should contribute to adaptation.

---

## Gated GRIT: Method

The proposed method introduces curvature-derived rank allocation into the GRIT optimization framework.

The overall procedure consists of:

1. Applying LoRA to selected projection matrices in the vision and language components of a vision-language model.
2. Collecting activation- and gradient-derived statistics during optimization.
3. Estimating Kronecker-factored curvature statistics through K-FAC.
4. Computing the eigenspectrum of the activation-side Fisher covariance matrix.
5. Converting the eigenvalues into normalized rank-wise importance weights.
6. Applying a minimum gate floor and renormalizing the resulting weights.
7. Incorporating the diagonal gate matrix into the low-rank update.

The following sections describe the mathematical formulation.

### 1. Fisher-Based Curvature Estimation

Let `A_f` denote the activation-side Fisher covariance matrix associated with a low-rank adaptation module:

```math
A_f \in \mathbb{R}^{r \times r}
```

Its eigendecomposition is computed as:

```math
A_f = V\Lambda V^\top
```

where:

```math
\Lambda = \mathrm{diag}(\lambda_1,\lambda_2,\ldots,\lambda_r)
```

Here:

- `V` contains the eigenvectors of the activation-side Fisher covariance.
- `Λ` is the diagonal matrix of eigenvalues.
- `λ_i` represents the curvature-derived importance associated with a rank direction.

The eigenspectrum provides a compact representation of the relative importance of directions within the low-rank adaptation subspace.

Directions associated with larger eigenvalues are treated as more important under the estimated local curvature structure.

### 2. Curvature-Derived Gate Computation

The eigenvalues are first converted into normalized importance weights.

For each rank direction `i`, the normalized gate value is:

```math
\hat{g}_i =
\frac{\lambda_i}
{\sum_{j=1}^{r}\lambda_j}
```

This normalization distributes the total gate allocation across the available rank directions.

To prevent complete suppression of any direction, a minimum gate floor `ε` is applied.

The final gate values are obtained through:

```math
g_i =
\frac{\max(\hat{g}_i,\epsilon)}
{\sum_{j=1}^{r}\max(\hat{g}_j,\epsilon)}
```

The resulting gate vector is:

```math
g = [g_1,g_2,\ldots,g_r]
```

The diagonal SuperGate matrix is then defined as:

```math
\Gamma =
\mathrm{diag}(g_1,g_2,\ldots,g_r)
```

In the reported implementation, the minimum SuperGate floor is:

```math
\epsilon = 0.04
```

The floor prevents any rank direction from being completely eliminated. This is important because the Fisher spectrum and the resulting gate values can change during training. A direction with relatively low importance at one point may become more relevant as optimization progresses.

The gates are derived from curvature statistics and do not constitute additional independently learned parameters.

### 3. Gated Low-Rank Updates

Given the diagonal SuperGate matrix `Γ`, the standard LoRA update is modified as:

```math
\Delta W = B\Gamma A
```

where:

```math
\Gamma = \mathrm{diag}(g_1,g_2,\ldots,g_r)
```

Expanding the gated update gives:

```math
\Delta W =
\sum_{i=1}^{r}g_i b_i a_i^\top
```

where:

- `b_i` is the `i`-th column of `B`.
- `a_i` is the `i`-th row of `A`.
- `g_i` is the curvature-derived importance weight for the corresponding rank direction.

This formulation allows individual rank components to contribute unequally to the final low-rank update.

Directions with larger gate values receive greater emphasis, while directions with smaller values remain active but contribute less.

The Fisher information therefore serves two related but distinct purposes in Gated GRIT:

1. **GRIT:** Fisher-based preconditioning incorporates curvature information into optimization.
2. **SuperGate:** The Fisher eigenspectrum determines the relative importance assigned to individual low-rank directions.

Together, these mechanisms provide curvature-aware optimization and curvature-derived rank allocation without increasing the number of trainable parameters.

### 4. Effective-Rank Analysis

The nominal LoRA rank specifies the maximum available adaptation capacity. However, it does not indicate how uniformly that capacity is used.

To examine rank utilization, Gated GRIT uses the entropy of the normalized Fisher eigenvalue spectrum.

First, define the normalized eigenvalue probabilities:

```math
p_i =
\frac{\lambda_i}
{\sum_j \lambda_j}
```

The effective rank is then calculated as:

```math
r_{\mathrm{eff}} =
\exp\left(
-\sum_i p_i\log p_i
\right)
```

The effective rank provides a measure of how broadly the adaptation capacity is distributed across the available directions.

- An effective rank closer to the nominal rank indicates broader utilization.
- A lower effective rank indicates that a smaller subset of directions accounts for a larger share of the allocation.

The effective-rank analysis is used to understand the rank utilization induced by curvature-derived gating. It is a diagnostic measure and does not represent hard pruning of LoRA components.

---

## Models and Datasets

### Vision-Language Models

The paper evaluates Gated GRIT using the following multimodal backbones:

- [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it)
- [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

The pretrained backbones are kept frozen while parameter-efficient adaptation is applied to selected projection layers.

### Dataset

The primary benchmark reported in the paper is:

**[VQA Online](https://huggingface.co/datasets/chongyenchen/VQAonline)**

The dataset contains visual questions and answers collected from online communities.

The reported train-validation-test split is:

| Split | Examples |
|---|---:|
| Training | 655 |
| Validation | 83 |
| Test | 83 |
| **Total** | **821** |

The paper evaluates the methods under the same dataset split and experimental protocol.

Additional experiments on image captioning and RefCOCO are part of the broader multimodal research direction and should be reported separately from the VQA Online results in the paper.

---

## Experimental Setup

### Methods Compared

The experiments compare:

- **LoRA:** Standard low-rank adaptation.
- **GRIT:** Curvature-aware low-rank optimization using K-FAC-based Fisher preconditioning and geometric reprojection.
- **Gated GRIT:** GRIT with curvature-derived rank-wise allocation through SuperGate.

All methods use identical target modules and matched parameter budgets for each rank configuration.

### LoRA Configuration

| Hyperparameter | Value |
|---|---|
| Adaptation rank | 4, 8 |
| LoRA alpha | 8 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj |
| Vision LoRA | Enabled |

### GRIT Configuration

| Hyperparameter | Value |
|---|---:|
| K-FAC damping | 0.005 |
| K-FAC update frequency | 150 |
| K-FAC minimum samples | 256 |
| Reprojection warmup | 200 |
| Rank adaptation start | 500 |
| Reprojection frequency | 300 |
| Two-sided reprojection | True |
| Curvature estimation | K-FAC |

### SuperGate Configuration

| Hyperparameter | Value |
|---|---|
| Gate source | Fisher spectrum |
| SuperGate floor | 0.04 |
| Additional trainable parameters | None |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Epochs | 3 |
| Learning rate | 2 × 10⁻⁶ |
| Gemma Gated GRIT learning rate | 2 × 10⁻⁶ |
| Precision | BF16 |
| Maximum sequence length | 1024 |
| Optimizer | AdamW |
| Warmup steps | 100 |

The exact method-specific configuration should be taken from the corresponding experiment notebook.

---

## Results

The following tables reproduce the main results reported in the paper on the VQA Online benchmark.

### Gemma 3 4B IT

| Method | Rank | Test Loss ↓ | ROUGE-L ↑ | METEOR ↑ | MoverScore ↑ | BERTScore ↑ |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 4 | 15.081 | 0.1577 | 0.1595 | 0.5355 | 0.5485 |
| GRIT | 4 | 15.636 | **0.1828** | **0.1634** | 0.5357 | **0.5922** |
| Gated GRIT | 4 | **9.438** | 0.1746 | 0.1304 | 0.5329 | 0.5789 |
| LoRA | 8 | 14.232 | 0.1570 | 0.1617 | 0.5345 | 0.5463 |
| GRIT | 8 | 15.069 | **0.1804** | **0.1634** | **0.5359** | **0.5892** |
| Gated GRIT | 8 | **9.793** | 0.1761 | 0.1632 | 0.5330 | 0.5872 |

### Llama 3.2 11B Vision

| Method | Rank | Test Loss ↓ | ROUGE-L ↑ | METEOR ↑ | MoverScore ↑ | BERTScore ↑ |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 4 | **12.889** | 0.1497 | **0.1504** | 0.5231 | **0.5042** |
| GRIT | 4 | 13.057 | 0.1906 | 0.1489 | 0.5274 | 0.4941 |
| Gated GRIT | 4 | 13.977 | **0.1947** | 0.1470 | **0.5279** | 0.5020 |
| LoRA | 8 | **11.964** | 0.1444 | **0.1524** | 0.5217 | **0.4964** |
| GRIT | 8 | 12.433 | **0.1951** | 0.1469 | **0.5276** | 0.4917 |
| Gated GRIT | 8 | 13.706 | 0.1918 | 0.1456 | 0.5271 | 0.4970 |

**Interpretation:** Gated GRIT achieves substantially lower test loss than the LoRA and GRIT baselines in the reported Gemma configurations, while maintaining competitive generation metrics. On Llama, Gated GRIT obtains the strongest ROUGE-L score at rank 4 and competitive BERTScore at rank 8. The results show that the effect of curvature-derived rank allocation varies across model architectures and evaluation metrics; it should not be interpreted as a uniform improvement across every metric.

---

## Gate and Effective-Rank Analysis

The paper further investigates how SuperGate distributes adaptation capacity across rank directions, model components, and backbone architectures.

### Effective Rank

The reported average effective ranks are:

| Model | Nominal Rank | Average Effective Rank |
|---|---:|---:|
| Gemma 3 4B | 4 | 2.50 |
| Gemma 3 4B | 8 | 3.59 |
| Llama 3.2 11B Vision | 4 | 2.89 |
| Llama 3.2 11B Vision | 8 | 4.05 |

Across all configurations, the effective rank remains below the nominal rank. This indicates that the adaptation capacity is not distributed uniformly across all available rank directions.

### Gate Distribution Statistics

The paper examines gate distributions using mean, standard deviation, minimum, maximum, and entropy.

#### Combined Vision-Language Modules

| Configuration | Mean | Std | Min | Max | Entropy |
|---|---:|---:|---:|---:|---:|
| Gemma R4 | 0.250 | 0.23857 | 0.04378 | 0.86864 | 6.17597 |
| Gemma R8 | 0.125 | 0.15796 | 0.03734 | 0.73855 | 6.73735 |
| Llama R4 | 0.250 | 0.22409 | 0.04589 | 0.86232 | 6.50457 |
| Llama R8 | 0.125 | 0.15878 | 0.04001 | 0.71519 | 7.02183 |

#### Vision Modules

| Configuration | Mean | Std | Min | Max | Entropy |
|---|---:|---:|---:|---:|---:|
| Gemma R4 | 0.250 | 0.18102 | 0.04916 | 0.71820 | 5.52421 |
| Gemma R8 | 0.125 | 0.12163 | 0.04536 | 0.61754 | 6.10561 |
| Llama R4 | 0.250 | 0.19768 | 0.04865 | 0.79988 | 5.87940 |
| Llama R8 | 0.125 | 0.14018 | 0.04311 | 0.65056 | 6.41737 |

#### Language Modules

| Configuration | Mean | Std | Min | Max | Entropy |
|---|---:|---:|---:|---:|---:|
| Gemma R4 | 0.250 | 0.27608 | 0.04378 | 0.86864 | 5.46180 |
| Gemma R8 | 0.125 | 0.18178 | 0.03734 | 0.73855 | 6.00729 |
| Llama R4 | 0.250 | 0.24791 | 0.04589 | 0.86232 | 5.74345 |
| Llama R8 | 0.125 | 0.17550 | 0.04001 | 0.71519 | 6.24002 |

The reported distributions are non-uniform across the evaluated configurations. A subset of rank directions receives larger gate values, while the remaining directions receive smaller values and remain active through the gate floor.

The paper also observes differences between vision and language components. Language modules generally exhibit higher peak gate values and greater variation, while vision modules show comparatively broader allocation in several configurations.

### Effective-Rank Utilization Statistics

| Metric | Gemma R4 | Gemma R8 | Llama R4 | Llama R8 |
|---|---:|---:|---:|---:|
| Average effective rank | 2.497 | 3.590 | 2.888 | 4.046 |
| Compression ratio | 0.312 | 0.449 | 0.361 | 0.506 |
| Sparsity (%) | 0.41 | 0.48 | 0.31 | 0.31 |

The paper reports that increasing nominal rank increases effective-rank utilization, while the resulting allocation remains selective rather than uniform.

### Entropy Across Model Depth

The paper also studies gate entropy across layers. The entropy of the normalized gate distribution is:

```math
H(g) = -\sum_{i=1}^{r}g_i\log(g_i)
```

Higher entropy indicates a broader distribution of gate values, whereas lower entropy indicates stronger concentration in a smaller subset of directions.

The reported minimum and maximum layer-wise entropy values are:

| Configuration | Minimum Entropy | Maximum Entropy | Range |
|---|---:|---:|---:|
| Gemma R4 | 0.54 | 1.26 | 0.72 |
| Gemma R8 | 1.08 | 1.89 | 0.81 |
| Llama R4 | 0.55 | 1.29 | 0.74 |
| Llama R8 | 1.16 | 1.89 | 0.73 |

The rank-8 configurations operate in a higher entropy regime than their rank-4 counterparts, indicating broader utilization of the available low-rank directions.

---

## Repository Organization

The experiments are notebook-based. A suggested organization for a cleaned public release is:

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

Adapt this structure to the actual files included in the repository. Do not commit model checkpoints, private access tokens, cached datasets, or large temporary training artifacts unless there is a specific reason to distribute them.

## Installation

A CUDA-capable environment is recommended. The notebooks use PyTorch, Hugging Face Transformers and PEFT, along with dataset and metric utilities.

A starting installation command is:

```bash
pip install torch torchvision
pip install transformers peft accelerate datasets
pip install numpy scikit-learn pillow requests
pip install rouge-score bert-score nltk pyemd
pip install git+https://github.com/AIPHES/emnlp19-moverscore.git
```

Exact package versions should be pinned in a release-specific `requirements.txt` after the environment used for the experiments has been frozen and tested.

### Hugging Face Authentication

Some model and dataset repositories require authentication or acceptance of access terms.

```bash
hf auth login
hf auth whoami
```

Do not hard-code a Hugging Face token in a notebook or commit it to GitHub. Use the environment's cached credentials or a secret-management mechanism.

## Running Experiments

1. Clone the repository and install the dependencies.
2. Configure Hugging Face access for the model and dataset.
3. Open the notebook corresponding to the desired method, model, and task.
4. Review the configuration, especially:
   - Model identifier
   - Dataset and split selection
   - Random seed
   - LoRA rank, alpha, dropout, and target modules
   - Batch size and gradient accumulation
   - Learning rate and number of epochs
   - K-FAC update and inversion settings
   - Reprojection frequency and warmup
   - SuperGate floor and rank-adaptation settings
   - Generation parameters and evaluation metrics
5. Run the notebook cells in order.
6. Preserve generated metrics, configuration, logs, predictions, and gate-analysis artifacts with the run identifier.

For large experiments, use a persistent terminal session manager such as `tmux` so training can continue if the SSH connection closes.

## Evaluation Metrics

The paper reports the following evaluation metrics:

- **Test loss:** Used to assess optimization quality and convergence behaviour.
- **ROUGE-L:** Lexical overlap based on the longest common subsequence.
- **METEOR:** A metric incorporating token alignment and linguistic matching.
- **MoverScore:** Embedding-based semantic similarity using contextualized representations and Earth Mover's Distance.
- **BERTScore:** Contextual semantic similarity based on pretrained language-model representations.

Values should only be compared when the same metric implementation, reference construction, text normalization, and generation settings are used.

## Limitations and Future Directions

The current study has several limitations:

- **Dependence on curvature estimation:** Gate allocation depends on the quality of the K-FAC Fisher approximation.
- **Limited evaluation scope:** The paper reports one VQA benchmark and two multimodal backbones. Broader evaluation across tasks and model scales is needed.
- **Dynamic rank allocation:** Fisher spectra and gate values can change during training. Further investigation of longer-term rank-allocation behaviour remains an open direction.
- **Theoretical understanding:** The precise relationship between Fisher spectra, effective rank, and downstream task performance is not yet fully understood.

Further work includes evaluating Gated GRIT across additional multimodal tasks, studying gate distributions at greater depth, and examining the relationship between curvature-derived allocation and downstream performance.

---

## Citation

If you use this work, please cite the Gated GRIT paper and the underlying GRIT work.

**Gated GRIT:**

```bibtex
@article{gatedgrit2026,
  title   = {Gated GRIT: Geometry-Aware Parameter-Efficient Fine-Tuning for Multimodal Vision-Language Models},
  author  = {Anonymous},
  year    = {2026}
}
```

**GRIT:**

```bibtex
@misc{saha2026grit,
  title         = {GRIT: Geometry-Aware PEFT with K-FAC Preconditioning, Fisher-Guided Reprojection, and Dynamic Rank Adaptation},
  author        = {Saha, Pritish and Rajbangshi, Chandrav and Goyal, Rudra and Goyal, Mohit and Deo, Anurag and Roy, Biswajit and Singh, Ningthoujam Dhanachandra and Goswami, Raxit and Das, Amitava},
  year          = {2026},
  eprint        = {2601.00231},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.00231}
}
```

The Gated GRIT citation should be updated with the final author list and publication details when the paper is publicly released.

## Acknowledgements

This work builds on the GRIT research direction and explores its extension to multimodal vision-language models. We acknowledge the authors and maintainers of the underlying models, datasets, libraries, and evaluation tools used in the experiments.
