<div align="center">
  <img src="docs/assets/lightmode.svg" alt="Evolution Logo" width="800"/>
  <br>
  <h1>Evolution Series: State of the Art Open Language Models</h1>
  <h3>2025 Update - Introducing Elevation, the new flagship model<h3>
</div>

Elevation is the third generation in the Evolution series of personal open-source language models, following the pioneering work of Embark and Emergence (both now deprecated). This research project was built upon the work of OLMo at AllenAI and uses all the professional methods for creating a state-of-the-art model. The model is made to excel in custom specialized fields based on its configuration and training.

## Research Motivation

The Evolution series is driven by a commitment to open research, reproducibility, and technical innovation in AI. Elevation explores novel architectures, data curation strategies, and optimization techniques, aiming to:

- Set new benchmarks for open LLM performance
- Enable transparent, reproducible model development
- Foster community-driven progress in AI research

## Methodology

Elevation employs a multi-stage training pipeline:

- **Stage 1:** Large-scale pretraining on diverse, high-volume datasets to capture broad linguistic and factual knowledge
- **Stage 2:** Targeted fine-tuning on curated, high-quality corpora to enhance model specialization and robustness
- **Model Averaging:** Ensemble and weight-averaging techniques to maximize generalization and stability

All training configurations, checkpoints, and data provenance are documented and released for scientific review.

## Training and Model Creation

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

## Acknowledgements

Elevation builds on the experience and lessons learned from Embark and Emergence, and is made possible by the support of the open-source AI research community.
