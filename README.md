# VRADAM Benchmarking Framework

This repository contains a comprehensive benchmarking framework for comparing the ADAM and VRADAM optimizers across various deep learning tasks.

## Project Structure

The project is organized as follows:

```
vam/
├── benchmarker.py              # Core benchmarking functionality
├── VRADAM.py                   # VRADAM optimizer implementation
├── architectures.py            # Common model architectures
├── text_datasets.py            # Text dataset utilities
├── diffusion_model.py          # Diffusion model implementation
├── sweep_vradam_cnn.py         # CNN sweep configuration
├── sweep_vradam_transformer.py # Transformer sweep configuration
├── sweep_vradam_gflownet.py    # GFlowNet sweep configuration
├── run_sweep_agent_cnn.py      # Script to run CNN sweeps
├── run_sweep_agent_transformer.py  # Script to run Transformer sweeps
├── run_sweep_agent_diffusion.py    # Script to run Diffusion sweeps
├── run_sweep_agent_gflownet.py     # Script to run GFlowNet sweeps
├── Edgeofstability/            # Edge of Stability experiments
├── requirements.txt            # Project dependencies
└── pyproject.toml              # Project configuration
```

## Benchmarks

The framework includes the following benchmarks:

1. **CNN for Image Classification**: Trains CNN models on image classification tasks.
2. **Transformer for Language Modeling**: Trains transformer models for language modeling.
3. **Diffusion Model**: Trains diffusion models for image generation.
4. **GFlowNet**: Trains GFlowNet models for generative modeling.

## Hyperparameter Sweeps

The framework integrates with Weights & Biases for hyperparameter optimization. For each benchmark, we can sweep:

- For Adam: learning rate and other relevant parameters
- For VRADAM: learning rate (eta), beta3 and lr_cutoff parameters

### Individual Sweep Commands

To run sweeps for specific models, use the following commands:

**CNN Sweep:**
```bash
python run_sweep_agent_cnn.py --optimizer_name VRADAM --model DeeperCNN --dataset CIFAR10 --count 1
```

**Diffusion Model Sweep:**
```bash
python run_sweep_agent_diffusion.py --optimizer VRADAM --count 1
```

**GFlowNet Sweep:**
```bash
python run_sweep_agent_gflownet.py --optimizer_name VRADAM --count 1
```

**Transformer Sweep:**
```bash
python run_sweep_agent_transformer.py --optimizer_name VRADAM --count 1
```

## Requirements

The major dependencies for this project are:

- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- NumPy (>=1.24.0)
- Matplotlib (>=3.7.0)
- Weights & Biases (>=0.15.0)
- NLTK (>=3.8.1)
- Gymnasium (>=0.28.0)
- tqdm (>=4.65.0)
- scikit-learn (>=1.2.0)
- scipy (>=1.10.0)
- PIL (>=9.0.0)
- transformers (>=4.30.0)
- datasets (>=2.11.0)
- seaborn (>=0.12.0)

See `requirements.txt` for the complete list of dependencies.

## Installation

To set up the environment and install all required dependencies:

```bash
pip install -r requirements.txt
```
