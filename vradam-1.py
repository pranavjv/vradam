"""
VRADAM: Velocity-Regularized Adam Optimizer

A PyTorch implementation of the VRADAM optimizer as described in the accompanying ICLR paper.
"""

import torch
from torch.optim import Optimizer


class VRADAM(Optimizer):
    """
    Velocity-Regularized Adam (VRADAM) optimizer.

    Combines Adam-like adaptive learning with velocity regularization and
    AdamW-style weight decay.

    Args:
        params: Iterable of parameters to optimize
        eta (float): Maximum learning rate (default: 0.001)
        beta1 (float): Coefficient for computing running average of gradient (default: 0.9)
        beta2 (float): Coefficient for computing running average of squared gradient (default: 0.999)
        beta3 (float): Velocity regularization coefficient (default: 1.0)
        eps (float): Term added for numerical stability (default: 1e-8)
        weight_decay (float): Weight decay coefficient (default: 0)
        power (float): Power for norm computation (default: 2)
        normgrad (bool): If True, compute norm on gradient; if False, on velocity (default: False)
        lr_cutoff (float): Controls minimal learning rate as eta/(lr_cutoff+1) (default: 10)
    """

    def __init__(
        self,
        params,
        eta: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 1.0,
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        power: float = 2.0,
        normgrad: bool = False,
        lr_cutoff: float = 10.0,
    ):
        defaults = dict(
            eta=eta,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            power=power,
            normgrad=normgrad,
            lr_cutoff=lr_cutoff,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta3 = group['beta3']
            eps = group['eps']
            weight_decay = group['weight_decay']
            power = group['power']
            eta = group['eta']
            normgrad = group['normgrad']
            lr_cutoff = group['lr_cutoff']

            # Compute velocity and second moment, accumulate norm
            total_sq_norm = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['sec_momentum_buffer'] = torch.zeros_like(p)

                state['step'] += 1

                # Update first moment (velocity)
                velocity = state['momentum_buffer']
                velocity.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Accumulate norm (on gradient or velocity)
                if normgrad:
                    total_sq_norm += float(grad.abs().pow(power).sum())
                else:
                    total_sq_norm += float(velocity.abs().pow(power).sum())

                # Update second moment
                sec_moment = state['sec_momentum_buffer']
                sec_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Compute adaptive learning rate
            lr = eta / (1 + min(beta3 * total_sq_norm, lr_cutoff))

            # Update parameters
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state['step']

                # Bias-corrected first and second moment estimates
                velocity_corrected = state['momentum_buffer'].div(1 - beta1 ** t)
                sec_moment_corrected = state['sec_momentum_buffer'].div(1 - beta2 ** t)

                # Compute update direction (in-place for efficiency)
                sec_moment_corrected.sqrt_().add_(eps)
                velocity_corrected.div_(sec_moment_corrected)

                # Apply weight decay and update
                p.mul_(1 - weight_decay * lr).add_(velocity_corrected, alpha=-lr)

        return loss
