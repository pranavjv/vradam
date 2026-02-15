# VRADAM: Velocity-Regularized Adam Optimizer

A PyTorch implementation of the VRADAM optimizer.

**Paper:** https://arxiv.org/abs/2505.13196v1

## Installation

Copy `vradam_clean.py` to your project.

## Usage

```python
from vradam_clean import VRADAM

optimizer = VRADAM(
    model.parameters(),
    eta=0.001,           # max learning rate
    beta1=0.9,           # momentum coefficient
    beta2=0.999,         # second moment coefficient
    beta3=1.0,           # velocity regularization strength
    weight_decay=0.01,   # weight decay (AdamW-style)
)
```

See [example.ipynb](example.ipynb) for a complete example.


## Citation

```bibtex
@misc{vaidhyanathan2025physicsinspiredoptimizervelocityregularized,
      title={A Physics-Inspired Optimizer: Velocity Regularized Adam}, 
      author={Pranav Vaidhyanathan and Lucas Schorling and Natalia Ares and Michael A. Osborne},
      year={2025},
      eprint={2505.13196},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.13196}, 
}
```
