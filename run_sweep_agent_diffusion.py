import argparse
import wandb
import json
import os
import sys
import torch
import numpy as np
from torchvision import transforms

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the diffusion model
from diffusion_model import EnhancedUNet, DiffusionModel
from VRADAM import VRADAM

class DiffusionSweepTrainer:
    def __init__(self, config):
        self.config = config
        
        # Import needed modules here to avoid import errors
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
        
        # Determine device
        if config.device == 'mps':
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            if not torch.backends.mps.is_available():
                print("MPS is not available. Using CPU instead.")
        elif config.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available. Using CPU instead.")
        else:
            self.device = torch.device('cpu')
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.generated_samples = []
        self.train_time = None
        
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        # Use smaller subset for sweep runs
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
        
        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        train_dataset = datasets.MNIST(
            root='data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Use full dataset for sweeps
        # Split into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        # Use fixed seed for consistent splits
        generator = torch.Generator().manual_seed(42)
        train_subset, val_set = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2)
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2)
        
    def setup_model(self):
        # Create EnhancedUNet model with attention if specified (with larger base channels)
        self.unet = EnhancedUNet(
            in_channels=1,  # MNIST is grayscale
            out_channels=1,  # Predict noise
            base_channels=self.config.unet_base_channels,
            time_emb_dim=self.config.unet_time_embed_dim,
            use_attention=self.config.use_attention if hasattr(self.config, 'use_attention') else True
        ).to(self.device)
        
        print(f"Using Enhanced UNet with base channels: {self.config.unet_base_channels}, "
              f"time embed dim: {self.config.unet_time_embed_dim}, "
              f"attention: {getattr(self.config, 'use_attention', True)}")
        
        # Create diffusion model wrapper
        self.diffusion = DiffusionModel(
            model=self.unet,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
            num_timesteps=self.config.num_timesteps,
            device=self.device
        )
        
        # Setup optimizer
        if self.config.optimizer == "VRADAM":
            # VRADAM specific parameters
            vradam_params = {
                'beta1': self.config.beta1,
                'beta2': self.config.beta2,
                'beta3': self.config.beta3 if hasattr(self.config, 'beta3') else 1.0,
                'eta': self.config.lr,
                'eps': self.config.eps,
                'weight_decay': self.config.weight_decay,
                'power': self.config.power if hasattr(self.config, 'power') else 2,
                'normgrad': self.config.normgrad if hasattr(self.config, 'normgrad') else True,
                'lr_cutoff': self.config.lr_cutoff if hasattr(self.config, 'lr_cutoff') else 19
            }
            self.optimizer = VRADAM(self.unet.parameters(), **vradam_params)
            print(f"Using VRADAM optimizer with eta={vradam_params['eta']}, beta3={vradam_params['beta3']}")
        elif self.config.optimizer == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.config.lr,
                'betas': (self.config.beta1, self.config.beta2),
                'eps': self.config.eps,
                'weight_decay': self.config.weight_decay
            }
            self.optimizer = torch.optim.Adam(self.unet.parameters(), **adam_params)
            print(f"Using Adam optimizer with lr={adam_params['lr']}")
        elif self.config.optimizer == "SGD":
            # SGD parameters
            sgd_params = {
                'lr': self.config.lr,
                'momentum': self.config.momentum if hasattr(self.config, 'momentum') else 0.9,
                'weight_decay': self.config.weight_decay,
                'nesterov': self.config.nesterov if hasattr(self.config, 'nesterov') else False
            }
            self.optimizer = torch.optim.SGD(self.unet.parameters(), **sgd_params)
            print(f"Using SGD optimizer with lr={sgd_params['lr']}, momentum={sgd_params['momentum']}")
        elif self.config.optimizer == "RMSPROP":
            # RMSprop parameters
            rmsprop_params = {
                'lr': self.config.lr,
                'alpha': self.config.alpha if hasattr(self.config, 'alpha') else 0.99,
                'eps': self.config.eps,
                'weight_decay': self.config.weight_decay,
                'momentum': self.config.momentum if hasattr(self.config, 'momentum') else 0.0
            }
            self.optimizer = torch.optim.RMSprop(self.unet.parameters(), **rmsprop_params)
            print(f"Using RMSprop optimizer with lr={rmsprop_params['lr']}, alpha={rmsprop_params['alpha']}")
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def train_epoch(self, epoch):
        self.unet.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
            
            # Calculate loss
            self.optimizer.zero_grad()
            loss = self.diffusion.p_losses(data, t)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        self.unet.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                batch_size = data.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                
                # Calculate loss
                loss = self.diffusion.p_losses(data, t)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def generate_samples(self, batch_size=4):
        self.unet.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(batch_size=batch_size, img_size=28)
        return samples
    
    def save_samples_grid(self, samples, epoch):
        """Save a grid of generated samples for visualization in wandb"""
        # Denormalize samples from [-1, 1] to [0, 1]
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        
        # Create a grid of images
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=int(np.sqrt(samples.shape[0])))
        
        # Convert to numpy for wandb
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))
        
        return wandb.Image(grid_np, caption=f"Epoch {epoch}")
    
    def run(self):
        """Run the benchmark and track metrics for the sweep"""
        import time
        start_time = time.time()
        
        # Initialize best validation loss for model saving
        best_val_loss = float('inf')
        best_model_path = None
        
        # Create directory for saving models
        os.makedirs("saved_models/diffusion", exist_ok=True)
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_filename = f"saved_models/diffusion/diffusion_{self.config.optimizer}_{wandb.run.id}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')}
                }, model_filename)
                best_model_path = model_filename
                print(f"Saved best model with val_loss={val_loss:.6f} to {model_filename}")
            
            # Generate samples at regular intervals or final epoch
            sample_interval = getattr(self.config, 'sample_every', 5)
            if epoch % sample_interval == 0 or epoch == self.config.epochs:
                samples = self.generate_samples(batch_size=16)
                samples_grid = self.save_samples_grid(samples, epoch)
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "samples": samples_grid,
                    "best_val_loss": best_val_loss
                })
            else:
                # Just log metrics without samples
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss
                })
        
        # Evaluate on test set
        test_loss = self.evaluate(self.test_loader)
        print(f"Final Test Loss: {test_loss:.6f}")
        
        # Generate final samples
        final_samples = self.generate_samples(batch_size=16)
        final_samples_grid = self.save_samples_grid(final_samples, self.config.epochs)
        
        self.train_time = time.time() - start_time
        
        # Log final metrics
        wandb.log({
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "test_loss": test_loss,
            "train_time": self.train_time,
            "final_samples": final_samples_grid,
            "best_val_loss": best_val_loss,
            "best_model_path": best_model_path
        })
        
        # Return validation loss as the main optimization metric
        # and track test loss as a secondary metric
        return self.val_losses[-1]  # Return final validation loss for optimization

def train_model(config=None):
    """
    Train diffusion model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Create and run benchmark
        trainer = DiffusionSweepTrainer(config)
        val_loss = trainer.run()
        
        # Return validation loss as the optimization metric
        return val_loss

def create_sweep_config(optimizer_type):
    """Create sweep configuration for diffusion model with specified optimizer type"""
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'final_val_loss',  # Optimize for validation loss
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'optimizer': {'value': optimizer_type},
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'},
            'seed': {'value': 42},
            'epochs': {'value': 50},  # More epochs for full dataset training
            'batch_size': {'value': 128},  # Allow larger batch sizes for faster training
            
            # Diffusion specific parameters - larger model
            'unet_base_channels': {'value': 96},  # Increased channel options
            'unet_time_embed_dim': {'value': 128},    # Larger time embeddings
            'num_timesteps': {'value':400},          # More diffusion steps
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            'use_attention': {'value': True},  # Always use attention for better quality
            'sample_every': {'value': 5},      # Save samples every 5 epochs
            
            # Common optimizer parameters
            'weight_decay': {'value': 1e-5},
        }
    }
    
    # Add optimizer-specific parameters
    if optimizer_type == 'ADAM':
        # For Adam, we sweep learning rate and standard Adam parameters
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'eps': {'value': 1e-8},
        })
    elif optimizer_type == 'VRADAM':
        # For VRADAM, we sweep learning rate, beta3, and lr_cutoff
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'beta3': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 2.0
            },
            'lr_cutoff': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 30
            },
            'eps': {'value': 1e-8},
            'power': {'value': 2},
            'normgrad': {'values': [True, False]}
        })
    elif optimizer_type == 'SGD':
        # For SGD, we sweep learning rate and momentum
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'momentum': {
                'value': 0.2 
            },
            'nesterov': {'value': True}
        })
    elif optimizer_type == 'RMSPROP':
        # For RMSprop, we sweep learning rate and alpha
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'alpha': {
                'value':0.1
            },
            'eps': {'value': 1e-8},
            'momentum': {
                'value':0.2
            }
        })
    
    return sweep_config

def run_sweep_agent(optimizer_name, count=10):
    """
    Create and run a new W&B sweep agent for a specific optimizer.
    
    Args:
        optimizer_name: Name of the optimizer ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')
        count: Number of runs to perform
    """
    # Ensure wandb is logged in
    wandb.login()
    
    print(f"\n{'='*60}")
    print(f"Running diffusion model sweeps on FULL MNIST dataset with {optimizer_name} optimizer")
    print(f"Using larger model architecture and optimizing for validation loss")
    print(f"{'='*60}\n")
    
    # Create a sweep configuration
    sweep_config = create_sweep_config(optimizer_name)
    
    # Define project name
    project_name = f"diffusion-mnist-{optimizer_name}"
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name
    )
    
    # Create a directory for saving sweep results
    results_dir = f"sweep_results_{optimizer_name}"
    os.makedirs(results_dir, exist_ok=True)
    config_path = f"{results_dir}/sweep_config_Diffusion.json"
    with open(config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
        
    print(f"Created new sweep with ID: {sweep_id} in project '{project_name}'")
    print(f"Sweep config saved to: {config_path}")
    
    # Run the sweep agent using the generalized train_model
    print(f"Starting sweep agent for {count} experiments (Sweep ID: {sweep_id}) ...")
    wandb.agent(sweep_id, function=train_model, count=count)
    print(f"Sweep agent finished for sweep ID: {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for diffusion model optimization")
    parser.add_argument("--optimizer", type=str, required=True, choices=["VRADAM", "ADAM", "SGD", "RMSPROP"],
                       help="Optimizer to run the sweep for ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    
    args = parser.parse_args()
    
    # Run the sweep agent
    sweep_id = run_sweep_agent(
        optimizer_name=args.optimizer,
        count=args.count
    ) 