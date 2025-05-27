import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train GFlowNet model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': 'GFlowNetModel',  # For GFlowNet tasks
            'dataset': 'GridWorld',  # Grid-based environment for GFlowNet
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': config.optimizer_name,  # 'VRADAM', 'ADAM', 'SGD', or 'RMSPROP'
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            
            # GFlowNet specific parameters
            'grid_size': config.grid_size,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'action_dim': config.action_dim,
            'state_dim': config.state_dim,
            'flow_matching_weight': config.flow_matching_weight,
            'reward_weight': config.reward_weight,
            'entropy_weight': config.entropy_weight,
            'reward_temp': config.reward_temp,
            
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
            params['weight_decay'] = config.sgd_weight_decay
            params['nesterov'] = config.sgd_nesterov
        elif config.optimizer_name == 'RMSPROP':
            params['lr'] = config.rmsprop_lr  # RMSprop's learning rate
            params['alpha'] = config.rmsprop_alpha
            params['eps'] = config.rmsprop_eps
            params['weight_decay'] = config.rmsprop_weight_decay
            params['momentum'] = config.rmsprop_momentum
        
        # Create and run benchmarker
        print(f"Running GFlowNet benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Log metrics to wandb
        wandb.log({
            "final_train_loss": results.get('final_train_loss', 0),
            "val_loss": results.get('val_losses', [])[-1] if results.get('val_losses', []) else float('inf'),
            "test_loss": results.get('test_loss', 0),
            "train_time": results.get('train_time', 0)
        })
        
        # GFlowNet specific metrics
        final_flow_match_loss = float('inf')
        final_reward_loss = float('inf')
        
        if 'flow_match_losses' in results and results['flow_match_losses']:
            final_flow_match_loss = results['flow_match_losses'][-1]
            wandb.log({"final_flow_match_loss": final_flow_match_loss})
            
        if 'reward_losses' in results and results['reward_losses']:
            final_reward_loss = results['reward_losses'][-1]
            wandb.log({"final_reward_loss": final_reward_loss})
            
        if 'entropy' in results and results['entropy']:
            wandb.log({"final_entropy": results['entropy'][-1]})
            
        if 'sample_diversity' in results:
            wandb.log({"sample_diversity": results.get('sample_diversity', 0)})
            
        if 'reward_mean' in results:
            wandb.log({
                "reward_mean": results.get('reward_mean', 0),
                "reward_max": results.get('reward_max', 0)
            })
        
        # Track training and validation curves
        for epoch in range(len(results.get('train_losses', []))):
            metrics = {
                "epoch": epoch,
                "train_loss": results['train_losses'][epoch],
            }
            
            # Add validation metrics if available
            if epoch < len(results.get('val_losses', [])):
                metrics["val_loss"] = results['val_losses'][epoch]
            
            # Add GFlowNet specific metrics
            if 'flow_match_losses' in results and epoch < len(results['flow_match_losses']):
                metrics["flow_match_loss"] = results['flow_match_losses'][epoch]
                
            if 'reward_losses' in results and epoch < len(results['reward_losses']):
                metrics["reward_loss"] = results['reward_losses'][epoch]
                
            if 'entropy' in results and epoch < len(results['entropy']):
                metrics["entropy"] = results['entropy'][epoch]
            
            wandb.log(metrics)
        
        # Use flow matching loss as the primary optimization metric
        # This better represents the quality of the GFlowNet's learning
        optimization_metric = final_flow_match_loss
        wandb.log({"optimization_metric": optimization_metric})
        
        # Also log reward metrics for tracking correlation
        wandb.log({
            "reward_loss_metric": final_reward_loss,
            "flow_match_reward_ratio": final_flow_match_loss / max(1e-8, abs(final_reward_loss))
        })
        
        return results

def create_vradam_sweep_config():
    """Create sweep configuration for VRADAM optimizer for GFlowNet."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower flow matching loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'VRADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 64},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # GFlowNet specific parameters
            'grid_size': {'value': 8},
            'hidden_dim': {'value': 128},
            'num_layers': {'value': 3},
            'action_dim': {'value': 5},  # 4 directions + stop
            'state_dim': {'value': 3},   # Channels representing grid state
            'flow_matching_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'reward_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'entropy_weight': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'reward_temp': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            
            # VRADAM parameters to optimize
            'eta': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},  # Base LR for VRADAM
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
    """Create sweep configuration for ADAM optimizer for GFlowNet."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower flow matching loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'ADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 64},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # GFlowNet specific parameters
            'grid_size': {'value': 8},
            'hidden_dim': {'value': 128},
            'num_layers': {'value': 3},
            'action_dim': {'value': 5},  # 4 directions + stop
            'state_dim': {'value': 3},   # Channels representing grid state
            'flow_matching_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'reward_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'entropy_weight': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'reward_temp': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            
            # ADAM specific parameters to optimize
            'adam_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},  # Base LR for ADAM
            'adam_beta1': {'value': 0.9},
            'adam_beta2': {'value': 0.999},
            'adam_weight_decay': {'value': 1e-5},
            'adam_eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_sgd_sweep_config():
    """Create sweep configuration for SGD optimizer for GFlowNet."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower flow matching loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'SGD'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 64},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # GFlowNet specific parameters
            'grid_size': {'value': 8},
            'hidden_dim': {'value': 128},
            'num_layers': {'value': 3},
            'action_dim': {'value': 5},  # 4 directions + stop
            'state_dim': {'value': 3},   # Channels representing grid state
            'flow_matching_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'reward_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'entropy_weight': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'reward_temp': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            
            # SGD specific parameters to optimize
            'sgd_lr': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
            'sgd_momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.99},
            'sgd_weight_decay': {'value': 1e-5},
            'sgd_nesterov': {'values': [True, False]},
        }
    }
    
    return sweep_config

def create_rmsprop_sweep_config():
    """Create sweep configuration for RMSprop optimizer for GFlowNet."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower flow matching loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'RMSPROP'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 64},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # GFlowNet specific parameters
            'grid_size': {'value': 8},
            'hidden_dim': {'value': 128},
            'num_layers': {'value': 3},
            'action_dim': {'value': 5},  # 4 directions + stop
            'state_dim': {'value': 3},   # Channels representing grid state
            'flow_matching_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'reward_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 10.0},
            'entropy_weight': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
            'reward_temp': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            
            # RMSprop specific parameters to optimize
            'rmsprop_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
            'rmsprop_alpha': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'rmsprop_eps': {'value': 1e-8},
            'rmsprop_weight_decay': {'value': 1e-5},
            'rmsprop_momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
        }
    }
    
    return sweep_config 