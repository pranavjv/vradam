import argparse
import wandb
import json
import os
# Import the specific config creators and the generalized train_model
from sweep_vradam_transformer import train_model, create_vradam_sweep_config, create_adam_sweep_config, create_sgd_sweep_config, create_rmsprop_sweep_config

def run_sweep_agent(optimizer_name, count=10):
    """
    Create and run a new W&B sweep agent for a specific optimizer on transformer language modeling.
    
    Args:
        optimizer_name: Name of the optimizer ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')
        count: Number of runs to perform
    """
    # Ensure wandb is logged in
    wandb.login()
    
    # Select the appropriate config function and set project name
    if optimizer_name == 'VRADAM':
        print(f"Creating new VRADAM sweep for transformer language modeling on WikiText2")
        sweep_config = create_vradam_sweep_config()
        project_name = "VRADAM-optimization-Transformer-WikiText2"
    elif optimizer_name == 'ADAM':
        print(f"Creating new ADAM sweep for transformer language modeling on WikiText2")
        sweep_config = create_adam_sweep_config()
        project_name = "ADAM-optimization-Transformer-WikiText2"
    elif optimizer_name == 'SGD':
        print(f"Creating new SGD sweep for transformer language modeling on WikiText2")
        sweep_config = create_sgd_sweep_config()
        project_name = "SGD-optimization-Transformer-WikiText2"
    elif optimizer_name == 'RMSPROP':
        print(f"Creating new RMSPROP sweep for transformer language modeling on WikiText2")
        sweep_config = create_rmsprop_sweep_config()
        project_name = "RMSPROP-optimization-Transformer-WikiText2"
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}. Choose 'VRADAM', 'ADAM', 'SGD', or 'RMSPROP'.")

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name,
    )
    
    # Save sweep configuration and ID locally
    results_dir = f"sweep_results_{optimizer_name}"
    os.makedirs(results_dir, exist_ok=True)
    config_path = f"{results_dir}/sweep_config_Transformer_WikiText2.json"
    with open(config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
        
    print(f"Created new sweep with ID: {sweep_id} in project '{project_name}'")
    print(f"Sweep config saved to: {config_path}")
    
    # Run the sweep agent
    print(f"Starting sweep agent for {count} experiments (Sweep ID: {sweep_id}) ...")
    wandb.agent(sweep_id, function=train_model, count=count)
    print(f"Sweep agent finished for sweep ID: {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for VRADAM or ADAM optimization on transformer language modeling")
    parser.add_argument("--optimizer_name", type=str, required=True, choices=["VRADAM", "ADAM", "SGD", "RMSPROP"],
                      help="Optimizer to run the sweep for ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    
    args = parser.parse_args()
    
    # Run the sweep agent
    sweep_id = run_sweep_agent(
        optimizer_name=args.optimizer_name,
        count=args.count
    ) 