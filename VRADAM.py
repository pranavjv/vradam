import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from torch.nn.utils import clip_grad_norm_
import time


class VRADAM(Optimizer):
    """
    Velocity-regularized ADAM (VRADAM) optimizer with Adam-like behavior and weight decay according to ADAMW
    """
    def __init__(self, params, beta1= 0.9, beta2= 0.999, beta3= 1, eta= 0.001, eps=1e-8, weight_decay=0, power=2, normgrad= True, lr_cutoff= 19):
        # eta corresponds to the maximal learning rate
        # if normgrad True the norm in the lr is is computed on the gradient, otherwise the velocity!
        # lr_cutoff controls the minimal learning rate, if = 19 minimal learning rate is eta/(19+1)
        defaults= dict(beta1 = beta1, beta2= beta2, beta3= beta3, eps = eps, weight_decay= weight_decay, power=power, eta= eta, normgrad= normgrad, lr_cutoff= lr_cutoff)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1= group['beta1']
            beta2= group['beta2']
            beta3= group['beta3']
            eps = group['eps']
            wd= group['weight_decay']
            power= group['power']
            eta= group['eta']
            normgrad= group['normgrad']
            lr_cutoff= group['lr_cutoff']

            # get velocity and second moment terms
            total_sq_norm = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                state= self.state[p]
                buf_vel= state.get('momentum_buffer', None)
                if buf_vel is None:
                    buf_vel = torch.zeros_like(p)
                    state['momentum_buffer'] = buf_vel
                buf_vel.mul_(beta1).add_(d_p, alpha=1- beta1)
                if  normgrad:
                    total_sq_norm += float(d_p.abs().pow(power).sum())
                else:
                    total_sq_norm += float(buf_vel.abs().pow(power).sum())

                buf_sec_mom= state.get('sec_momentum_buffer', None)
                if buf_sec_mom is None:
                    buf_sec_mom = torch.zeros_like(p)
                    state['sec_momentum_buffer'] = buf_sec_mom
                buf_sec_mom.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)


                _t= state.get('step', None)
                if _t is None:
                    state['step'] = 0
                state['step'] += 1
                t= state['step']



            lr= eta/(1 + min(float(beta3 * total_sq_norm), float(lr_cutoff)))
            # Commenting out the print to avoid flooding output during benchmarking
            # print(f"total_sq_norm: {total_sq_norm}")
            for p in group['params']:
                if p.grad is None:
                    continue
                buf_vel= self.state[p]['momentum_buffer']
                buf_sec_mom= self.state[p]['sec_momentum_buffer']

                # could be done more efficiently in place
                # rescale  velocity and second moment terms


                tmp1= buf_vel.div(1- beta1**t)
                tmp2= buf_sec_mom.div(1- beta2**t)

                
                tmp2.sqrt_().add_(eps)
                tmp1.div_(tmp2)

                # update parameter with weight decay
                p.mul_(1-wd*lr).add_(tmp1, alpha=-lr)
        return loss