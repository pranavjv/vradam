import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train transformer model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': 'TransformerModel',  # For transformer tasks
            'dataset': 'WikiText2',  # Language modeling dataset
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': config.optimizer_name,  # 'VRADAM' or 'ADAM'
            'batch_size': config.batch_size,
            'max_seq_len': config.max_seq_len,
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'epochs': config.epochs,
            
            # Scheduler params
            'lr_scheduler_type': config.lr_scheduler_type,
            'lr_warmup_epochs': config.lr_warmup_epochs,
            'lr_warmup_factor': config.lr_warmup_factor,
            'lr_eta_min': config.lr_eta_min,
            
            # Set fixed seed for consistent initialization
            'seed': config.seed
        }

        # Add optimizer specific parameters
        if config.optimizer_name == 'VRADAM':
            params['eta'] = config.eta  # VRADAM's learning rate
            params['beta1'] = config.beta1
            params['beta2'] = config.beta2
            params['beta3'] = config.beta3
            params['power'] = config.power
            params['normgrad'] = config.normgrad
            params['lr_cutoff'] = config.lr_cutoff
            params['weight_decay'] = config.weight_decay
            params['eps'] = config.eps
        elif config.optimizer_name == 'ADAM':
            params['lr'] = config.adam_lr  # Adam's learning rate
            params['beta1'] = config.adam_beta1
            params['beta2'] = config.adam_beta2
            params['weight_decay'] = config.adam_weight_decay
            params['eps'] = config.adam_eps
        elif config.optimizer_name == 'SGD':
            params['lr'] = config.sgd_lr  # SGD's learning rate
            params['momentum'] = config.sgd_momentum
            #params['weight_decay'] = config.sgd_weight_decay
            params['nesterov'] = config.sgd_nesterov
        elif config.optimizer_name == 'RMSPROP':
            params['lr'] = config.rmsprop_lr  # RMSprop's learning rate
            params['alpha'] = config.rmsprop_alpha
            params['eps'] = config.rmsprop_eps
            #params['weight_decay'] = config.rmsprop_weight_decay
            params['momentum'] = config.rmsprop_momentum
        
        # Create and run benchmarker
        print(f"Running transformer benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Get training, validation, and test perplexity
        final_train_ppl = results.get('final_train_perplexity', 0)
        val_ppl = results.get('val_perplexities', [])[-1] if results.get('val_perplexities', []) else float('inf')
        test_ppl = results.get('test_perplexity', 0)
        
        # Log metrics to wandb for language modeling tasks
        wandb.log({
            "final_train_loss": results.get('final_train_loss', 0),
            "final_train_perplexity": final_train_ppl,
            "val_perplexity": val_ppl,
            "test_loss": results.get('test_loss', 0),
            "test_perplexity": test_ppl,
            "train_time": results.get('train_time', 0)
        })
        
        # Track training and validation curves
        for epoch in range(len(results.get('train_losses', []))):
            metrics = {
                "epoch": epoch,
                "train_loss": results['train_losses'][epoch],
                "train_perplexity": results['train_perplexities'][epoch] if epoch < len(results.get('train_perplexities', [])) else None
            }
            
            # Add validation metrics if available
            if epoch < len(results.get('val_losses', [])):
                metrics["val_loss"] = results['val_losses'][epoch]
            if epoch < len(results.get('val_perplexities', [])):
                metrics["val_perplexity"] = results['val_perplexities'][epoch]
            
            wandb.log(metrics)
        
        # For language modeling, lower perplexity is better and W&B minimizes by default
        # Use validation perplexity as the optimization metric (best practice)
        optimization_metric = val_ppl
        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_vradam_sweep_config():
    """Create sweep configuration for VRADAM optimizer for transformer language modeling."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation perplexity is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'VRADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},  # Use small dataset for faster iterations
            'epochs': {'value': 100},  # Transformer training can be slow
            'batch_size': {'value': 32},
            
            # Reproducibility
            'seed': {'value': 0},  # Fixed seed for consistent model initialization
            
            # Scheduler Parameters
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # Transformer specific parameters
            'max_seq_len': {'value': 64},
            'embed_dim': {'value': 128},
            'hidden_dim': {'value': 256},
            
            # VRADAM parameters to optimize
            'eta': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.1},  # Base LR for VRADAM
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'beta3': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            'power': {'value': 2},
            'normgrad': {'values': [True, False]},
            'lr_cutoff': {'distribution': 'int_uniform', 'min': 5, 'max': 30},
            'weight_decay': {'value': 1e-5},
            'eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_adam_sweep_config():
    """Create sweep configuration for ADAM optimizer for transformer language modeling."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation perplexity is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'ADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},  # Use small dataset for faster iterations
            'epochs': {'value': 100},  # Transformer training can be slow
            'batch_size': {'value': 32},
            
            # Reproducibility
            'seed': {'value': 0},  # Fixed seed for consistent model initialization
            
            # Scheduler Parameters
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # Transformer specific parameters
            'max_seq_len': {'value': 64},
            'embed_dim': {'value': 128},
            'hidden_dim': {'value': 256},
            
            # ADAM specific parameters to optimize
            'adam_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.1},  # Base LR for ADAM
            'adam_beta1': {'value': 0.9},
            'adam_beta2': {'value': 0.999},
            'adam_weight_decay': {'value': 1e-5},
            'adam_eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_sgd_sweep_config():
    """Create sweep configuration for SGD optimizer for transformer language modeling."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation perplexity is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'SGD'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100},
            'batch_size': {'value': 32},
            
            # Reproducibility
            'seed': {'value': 0},
            
            # Scheduler Parameters
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # Transformer specific parameters
            'max_seq_len': {'value': 64},
            'embed_dim': {'value': 128},
            'hidden_dim': {'value': 256},
            
            # SGD specific parameters to optimize
            'sgd_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},  # SGD typically needs higher LR
            'sgd_momentum': {'value':0.2},
            'sgd_nesterov': {'value': True},
        }
    }
    
    return sweep_config

def create_rmsprop_sweep_config():
    """Create sweep configuration for RMSprop optimizer for transformer language modeling."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation perplexity is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'RMSPROP'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100},
            'batch_size': {'value': 32},
            
            # Reproducibility
            'seed': {'value': 0},
            
            # Scheduler Parameters
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # Transformer specific parameters
            'max_seq_len': {'value': 64},
            'embed_dim': {'value': 128},
            'hidden_dim': {'value': 256},
            
            # RMSprop specific parameters to optimize
            'rmsprop_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.1},
            'rmsprop_alpha': {'value':0.1},
            'rmsprop_eps': {'value': 1e-8},
            'rmsprop_momentum': {'value':0.2},
        }
    }
    
    return sweep_config 