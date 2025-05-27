import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': config.model,
            'dataset': config.dataset,
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': config.optimizer_name, # 'VRADAM' or 'ADAM'
            'batch_size': config.batch_size,
            'max_seq_len': config.max_seq_len,
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'epochs': config.epochs,
            
            # Common Scheduler params from sweep config
            'lr_scheduler_type': config.lr_scheduler_type,
            'lr_warmup_epochs': config.lr_warmup_epochs,
            'lr_warmup_factor': config.lr_warmup_factor,
            'lr_eta_min': config.lr_eta_min
            # Note: 'seed' could be added here if we sweep it or fix it per sweep
        }

        if config.optimizer_name == 'VRADAM':
            params['lr'] = config.eta # VRADAM's main LR, Benchmarker expects 'lr'
            params['eta'] = config.eta
            params['beta1'] = config.beta1
            params['beta2'] = config.beta2
            params['beta3'] = config.beta3
            params['power'] = config.power
            params['normgrad'] = config.normgrad
            params['lr_cutoff'] = config.lr_cutoff
            params['weight_decay'] = config.weight_decay
            params['eps'] = config.eps
        elif config.optimizer_name == 'ADAM':
            params['lr'] = config.adam_lr # ADAM's LR, Benchmarker expects 'lr'
            # Benchmarker's Adam setup expects individual beta1, beta2 and forms the tuple
            params['beta1'] = config.adam_beta1 
            params['beta2'] = config.adam_beta2
            params['weight_decay'] = config.adam_weight_decay
            params['eps'] = config.adam_eps
        elif config.optimizer_name == 'SGD':
            params['lr'] = config.sgd_lr
            params['momentum'] = config.sgd_momentum
            params['nesterov'] = config.sgd_nesterov
            # Assuming Benchmarker handles weight_decay for SGD if provided in params
            if hasattr(config, 'sgd_weight_decay'):
                params['weight_decay'] = config.sgd_weight_decay
        elif config.optimizer_name == 'RMSPROP':
            params['lr'] = config.rmsprop_lr
            params['alpha'] = config.rmsprop_alpha
            params['eps'] = config.rmsprop_eps
            params['momentum'] = config.rmsprop_momentum
            # Assuming Benchmarker handles weight_decay for RMSPROP if provided in params
            if hasattr(config, 'rmsprop_weight_decay'):
                params['weight_decay'] = config.rmsprop_weight_decay
        else:
            raise ValueError(f"Unsupported optimizer_name in sweep config: {config.optimizer_name}")
        
        # Create and run benchmarker
        print(f"Running benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Log metrics to wandb
        # Track key metrics based on model type
        if benchmark.task_type == "language_modeling":
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_perplexity": results['final_train_perplexity'],
                "test_loss": results['test_loss'],
                "test_perplexity": results['test_perplexity'],
                "train_time": results['train_time']
            })
            
            # Track training curves
            for epoch, (loss, perplexity) in enumerate(zip(results['train_losses'], results['train_perplexities'])):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_perplexity": perplexity
                })
                
        else:  # Classification tasks
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_acc": results['final_train_acc'],
                "final_val_loss": results['val_losses'][-1] if results['val_losses'] else None,
                "final_val_acc": results['val_accs'][-1] if results['val_accs'] else None,
                "test_loss": results['test_loss'],
                "test_acc": results['test_acc'],
                "train_time": results['train_time']
            })
            
            # Track training curves
            for epoch, (loss, acc) in enumerate(zip(results['train_losses'], results['train_accs'])):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_acc": acc
                })
        
        # Determine the optimization metric based on task
        if benchmark.task_type == "language_modeling":
            # For language modeling, lower perplexity is better
            optimization_metric = results['test_perplexity']
        else:
            # For classification, higher accuracy is better, but W&B minimizes by default
            # so we negate the accuracy
            #optimization_metric = -results['test_acc']
            optimization_metric = results['val_losses'][-1]

        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_vradam_sweep_config(model_type, dataset):
    """Create sweep configuration for VRADAM optimizer."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'VRADAM'},
            
            # Fixed parameters (shared across optimizers)
            'model': {'value': model_type},
            'dataset': {'value': dataset},
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100},
            'batch_size': {'value': 1024},
            
            # Scheduler Parameters (Fixed for this sweep)
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # VRADAM specific parameters to optimize (or fixed as per user edits)
            'eta': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 0.1}, # Base LR for VRADAM
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'beta3': {'distribution': 'uniform', 'min': 0.1, 'max': 1.5},
            'power': {'value': 2},
            'normgrad': {'values': [True, False]},
            'lr_cutoff': {'distribution': 'int_uniform', 'min': 3, 'max': 30},
            'weight_decay': {'value': 1e-5},
            'eps': {'value': 1e-8},
        }
    }
    
    # Add model-specific parameters (common)
    if model_type in ['TransformerModel', 'MLPModel']:
        sweep_config['parameters'].update({
            'embed_dim': {'values': [128, 256, 384, 512]},
            'hidden_dim': {'values': [256, 512, 768, 1024]},
            'max_seq_len': {'values': [64, 128, 256]},
        })
    else: # SimpleCNN defaults
        sweep_config['parameters'].update({
            'embed_dim': {'value': 300},
            'hidden_dim': {'value': 512},
            'max_seq_len': {'value': 256},
        })
    
    return sweep_config

def create_adam_sweep_config(model_type, dataset):
    """Create sweep configuration for ADAM optimizer."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize' 
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'ADAM'},
            
            # Fixed parameters (shared across optimizers)
            'model': {'value': model_type},
            'dataset': {'value': dataset},
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100},
            'batch_size': {'value': 1024},
            
            # Scheduler Parameters (Fixed, same as VRADAM sweep)
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},
            
            # ADAM specific parameters to optimize (only LR here, others fixed)
            'adam_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}, # Base LR for ADAM
            'adam_beta1': {'value': 0.9},
            'adam_beta2': {'value': 0.999},
            'adam_weight_decay': {'value': 1e-5}, # Using same fixed value as VRADAM sweep
            'adam_eps': {'value': 1e-8},      # Using same fixed value as VRADAM sweep
        }
    }
    
    # Add model-specific parameters (common)
    sweep_config['parameters'].update({
        'embed_dim': {'value': 300},
        'hidden_dim': {'value': 512},
        'max_seq_len': {'value': 256},
    })
    
    return sweep_config

def create_sgd_sweep_config(model_type, dataset):
    """Create sweep configuration for SGD optimizer."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'SGD'},
            
            # Model and Dataset
            'model': {'value': model_type},
            'dataset': {'value': dataset},
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100}, # Consistent epochs
            'batch_size': {'value': 1024}, # Consistent batch size
            
            # Scheduler Parameters (Copying from ADAM/VRADAM for consistency)
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5}, # SGD might need different eta_min, adjust if necessary
            
            # SGD specific parameters
            'sgd_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'sgd_momentum': {'value': 0.9},
            'sgd_nesterov': {'value': True},
            'sgd_weight_decay': {'value': 1e-5}, # Adding a common weight decay, can be swept too
        }
    }
    # Add common model-agnostic parameters (placeholders for CNNs)
    sweep_config['parameters'].update({
        'embed_dim': {'value': 300},
        'hidden_dim': {'value': 512},
        'max_seq_len': {'value': 256},
    })
    return sweep_config

def create_rmsprop_sweep_config(model_type, dataset):
    """Create sweep configuration for RMSprop optimizer."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'RMSPROP'},

            # Model and Dataset
            'model': {'value': model_type},
            'dataset': {'value': dataset},
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 100}, # Consistent epochs
            'batch_size': {'value': 1024}, # Consistent batch size

            # Scheduler Parameters
            'lr_scheduler_type': {'value': 'WarmupCosineAnnealing'},
            'lr_warmup_epochs': {'value': 5},
            'lr_warmup_factor': {'value': 0.1},
            'lr_eta_min': {'value': 1e-5},

            # RMSprop specific parameters
            'rmsprop_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.1},
            'rmsprop_alpha': {'value': 0.99}, # User specified
            'rmsprop_eps': {'value': 1e-8},   # User specified
            'rmsprop_momentum': {'value': 0.2}, # User specified
            'rmsprop_weight_decay': {'value': 1e-5}, # Adding a common weight decay
        }
    }
    # Add common model-agnostic parameters (placeholders for CNNs)
    sweep_config['parameters'].update({
        'embed_dim': {'value': 300},
        'hidden_dim': {'value': 512},
        'max_seq_len': {'value': 256},
    })
    return sweep_config

def run_sweeps():
    """Run sweeps for specified optimizers, models, and datasets."""
    # List of optimizers to run sweeps for
    optimizer_types = ['VRADAM', 'ADAM', 'SGD', 'RMSPROP']
    
    # List of model and dataset combinations to benchmark
    model_dataset_pairs = [
        ('DeeperCNN', 'CIFAR10'),
        ('SimpleCNN', 'CIFAR10'),
        # ('MLPModel', 'IMDB'),
        # ('TransformerModel', 'WikiText2')
    ]
    
    # Set up wandb project
    wandb.login()
    
    all_sweep_ids = {} # Changed from list to dict to store by project_name
    for optimizer_type in optimizer_types:
        print(f"\n===== Starting Sweeps for Optimizer: {optimizer_type} =====")
        for model_type, dataset in model_dataset_pairs:
            # Define project name dynamically
            project_name = f"{optimizer_type}-{dataset}"
            print(f"\nSetting up sweep for {optimizer_type} - {model_type} on {dataset} in project {project_name}")
            
            # Select the appropriate config function
            if optimizer_type == 'VRADAM':
                sweep_config = create_vradam_sweep_config(model_type, dataset)
            elif optimizer_type == 'ADAM':
                sweep_config = create_adam_sweep_config(model_type, dataset)
            elif optimizer_type == 'SGD':
                sweep_config = create_sgd_sweep_config(model_type, dataset)
            elif optimizer_type == 'RMSPROP':
                sweep_config = create_rmsprop_sweep_config(model_type, dataset)
            else:
                print(f"Skipping unknown optimizer type: {optimizer_type}")
                continue
            
            # Create and run the sweep
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            
            # Store sweep ID
            if project_name not in all_sweep_ids:
                all_sweep_ids[project_name] = []
            all_sweep_ids[project_name].append(sweep_id)
            
            print(f"Created sweep with ID: {sweep_id} for {project_name}")
            print(f"Run wandb agent: wandb agent {wandb.wandb_sdk.lib.credentials.get_entity()}/{project_name}/{sweep_id}")

    # Save all sweep IDs to a JSON file
    # Create directory if it doesn't exist
    sweep_dir = "sweep_results"
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_ids_path = os.path.join(sweep_dir, f"sweep_ids_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(sweep_ids_path, 'w') as f:
        json.dump(all_sweep_ids, f, indent=4)
    print(f"\nAll sweep IDs saved to {sweep_ids_path}")
    print("You can now run 'python run_sweep_agent.py <sweep_id>' or use the printed wandb agent commands.")

if __name__ == "__main__":
    run_sweeps() 