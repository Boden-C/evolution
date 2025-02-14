<div align="center">
  <img src="docs/assets/lightmode.svg" alt="Evolution Logo" width="800"/>
  <br>
  <h1>Evolution Series: State of the Art Open Language Models</h1>
  <h3>2025 Update - Introducing Elevation, the new flagship model</h3>
</div>

Elevation is the third generation in the Evolution series of personal open-source language models, following the pioneering work of Embark and Emergence (both now deprecated). This research project was built upon the work of OLMo at AllenAI and uses professional methods for creating a state-of-the-art model. The model is designed to excel in custom specialized fields based on its configuration and training.

## Research Motivation

The Evolution series is driven by a commitment to open research, reproducibility, and technical innovation in AI. The 3 main goals are:

-   Create a more customizable model that can excel in specialized fields
-   Produce a cheap, tiny model that can easily accomplish specific tasks compared to popular models
-   Research and learn about open-source LLM models

## Design

Elevation employs a multi-stage training pipeline.

-   **Stage 1:** Large-scale pretraining on diverse, high-volume datasets to capture broad linguistic and factual knowledge
-   **Stage 2:** Targeted fine-tuning on curated, high-quality corpora to enhance model specialization and robustness
-   **Model Averaging:** Multiple independent fine-tuning trajectories can be aggregated by weight averaging ("model soups") to produce a single, robust checkpoint. Uniform and weighted soups with coefficients $\alpha_i$ satisfying $\sum_i \alpha_i=1$ are supported conceptually. The implementation aligns state-dicts, handles compatible shape variations, can optionally apply layer-wise clipping, and perform post-average calibration (e.g. layer norm / logit scale) where implemented.

### Optimization Techniques

Autoregressive decoding stores per-layer K/V tensors to avoid re-computation of prefix attention; cached tensors may be sharded across devices for memory balance and tiled for fast broadcast in adapter runtimes. The generation runtime aims to expose APIs for incremental append and batched gather.

Kernel selection is guided by compute capability and context size: tensor-core FP16/BF16 matmuls, cuBLASLt heuristics, and (where present) Triton custom kernels for fused attention or quantization paths. Tuned block sizes follow internal benchmarks. Fallbacks to numerically stable reference kernels are retained when specialized paths are unsuitable.

FSDP shards parameters and optimizer state across ranks to reduce per-device memory and enable larger global batch sizes. Configuration covers parameter flattening, backward prefetch, activation checkpointing/offload, and compatibility with weight-tying and state-dict IO under sharding.

$$
M_{local} \approx \frac{M_{params}}{N} + M_{activations\_local} + \text{(grad + opt state)/}N + O(\text{overhead})
$$

Training uses AMP (`torch.cuda.amp.autocast`) with dynamic loss scaling (`torch.cuda.amp.GradScaler`) when FP16 is employed; BF16 typically omits dynamic scaling. The training loop scales the loss by $S$ before backward, unscales gradients for clipping and optimizer steps, and adapts $S$ via overflow counters.

Training minimizes next-token negative log-likelihood augmented by weight decay and optional label smoothing $\epsilon$ for calibration. Per-token masking, adaptive smoothing schedules, and L2 regularization are combined into the loss for robust generalization.

$$
L = -(1-\epsilon)\sum_i y_i \log \hat y_i - \epsilon \sum_i \frac{1}{V}\log \hat y_i + \lambda \|W\|_2^2
$$

### Architectural Features

Input token embeddings and output projection share parameters ($W_{out}=W_{embed}^\top$) to reduce parameter count and improve calibration; tied tensors require coordinated handling under sharding so the state dict remains consistent.

RMSNorm replaces LayerNorm where beneficial; it normalizes by root mean square and can be implemented as a fused CUDA kernel for memory and speed advantages. For vector $x$ of dim $d$:

$$
\mathrm{RMS}(x)=\sqrt{\tfrac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}, \quad y=g\odot \frac{x}{\mathrm{RMS}(x)}
$$

Relative position biases may be added to attention logits as learned or bucketing biases $b_{i-j}$ so the model generalizes to shifted contexts; implementations can include T5-style additive biases or low-rank decompositions. A fused attention kernel can integrate the bias lookup into the QK scaling path.

$$
e_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + b_{i-j}
$$

ALiBi injects linear bias slopes into attention logits to favor recency (slopes are typically non-positive so distant tokens are down-weighted), while RoPE rotates query/key subspaces to encode relative phase; both can be implemented as fused ops with distance bucketing and slope schedules for long contexts.

$$
\text{ALiBi: } e_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + m_h (i-j) \\
\text{RoPE: rotate}(x_{2k:2k+2}) = \begin{bmatrix}\cos\theta_k & -\sin\theta_k\\ \sin\theta_k & \cos\theta_k\end{bmatrix} x_{2k:2k+2}
$$

### Performance Improvements

Blockwise (flash) attention kernels reduce peak memory use by avoiding materialization of the full $T \times T$ score matrix, moving from $O(T^2)$ memory toward $O(T \cdot d_{head})$ (streamed blocks plus workspace). Flash kernels are preferred when sequence lengths and hardware satisfy kernel heuristics; see `src/runtime/train.py` if present.

Residual and embedding dropout regularize training with schedule-aware rates and optional variational variants; when available, dropout masks may be fused into attention/MLP kernels to avoid extra memory traffic where $\tilde x = x \odot \mathrm{Bernoulli}(1-p)/(1-p)$.

SwiGLU gating uses two linear branches and a SiLU gating nonlinearity to increase expressivity: output $=W_o((W_a x)\odot \mathrm{SiLU}(W_b x))$. Fused kernels (where implemented) amortize memory for the extra branch and improve tensor-core utilization; see `src/model.py` and `src/config.py`.

$$
\mathrm{SwiGLU}(x) = W_o\big((W_a x) \odot \mathrm{SiLU}(W_b x)\big)
$$

### Data Handling

Batches currently use right-padding by default for simplicity and generation alignment. Packed batching is supported conceptually; masks propagate into attention/softmax kernels to minimize wasted computation. Alternative left-padding modes can be added for runtimes that prefer contiguous prefill.

We apply hierarchical shuffling (file-level then buffer-level) with deterministic seed trees per epoch and rank to ensure reproducibility while enabling controlled perturbations for soups. The shuffling layer is part of the streaming data pipeline.

### Evaluation and Quantization

We continuously evaluate on curated validation slices (domain, length, difficulty) to track specialization and overfitting. Subset evaluation can run lightweight metrics on validation nodes with multiple seeds and contribute to early-stopping heuristics and soup weighting.

GPTQ performs low-bit per-channel or per-block quantization that approximately minimizes layerwise reconstruction error using second-order (Hessian-aware) criteria. Scales (and optional zero points) are stored per block alongside quantized weight integers. The simplified scalar formula below illustrates uniform quantization, not the full Hessian-aware procedure:

$$
Q(w)=\text{round}\!\left(\frac{w}{s}\right)\cdot s, \quad s=\arg\min_s \|W - s\cdot Q(W)\|_F
$$

## Model Creation

To train and build Elevation from source:

1. Install [PyTorch](https://pytorch.org) and required dependencies for your system.
2. Clone the Evolution repository:
    ```bash
    git clone https://github.com/YOUR_ORG/Evolution.git
    cd Evolution
    pip install -e .[all]
    ```
3. Prepare your training data according to the provided configuration templates.
4. Launch training using the provided scripts and configuration files:
    ```bash
    torchrun --nproc_per_node=8 scripts/train.py configs/elevation-stage1.yaml
    # For fine-tuning:
    torchrun --nproc_per_node=8 scripts/train.py configs/elevation-stage2.yaml
    ```
5. All checkpoints and model artifacts will be saved locally for further analysis and deployment.

## Citation

If you use Elevation or the Evolution series in your research, please cite this project:

```bibtex
@misc{evolution2025elevation,
   title={Elevation: The Evolution Series State-of-the-Art Language Model},
   author={Evolution Contributors},
   year={2025},
   url={https://github.com/Boden-C/evolution},
}
```

This project builds on the OLMo effort. Please also cite their work:

```bibtex
@misc{olmo20242olmo2furious,
   title={2 OLMo 2 Furious},
   author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
   year={2024},
   eprint={2501.00656},
   archivePrefix={arXiv},
   primaryClass={cs.CL},
   url={https://arxiv.org/abs/2501.00656},
}
```

## Implementation Status & Notes

The README describes design intent. Some features (fused kernels, advanced soup calibration, fully integrated generation APIs, fused dropout, comprehensive kernel auto-tuning) may be partially implemented or pending. Statements about performance assume correct hardware support and kernel availability. Quantization details beyond the simplified formula rely on dedicated calibration passes.

## Acknowledgements

Elevation builds on the experience and lessons learned from Embark and Emergence, and is made possible by the support of the open-source AI research community.
