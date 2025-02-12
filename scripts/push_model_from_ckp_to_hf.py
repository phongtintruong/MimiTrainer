import torch, os
from mimitransformers import MeomeoModel, MeomeoConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

def load_generator_only(path, generator):
    """
    Load only the generator model from the checkpoint.

    Parameters:
    - path: Path to the checkpoint file (e.g., 'MimiTrainer_00000100').
    - generator: The generator model to load the state_dict into.

    Returns:
    - The generator model with the loaded state_dict.
    """
    # Use os.path.exists to check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    # Load the checkpoint
    pkg = torch.load(path, map_location='cpu')
    print(pkg.keys())

    # Check if the generator state_dict exists in the checkpoint
    if 'generator' not in pkg:
        raise KeyError("The checkpoint does not contain the generator state_dict.")

    # Load the generator state_dict
    generator.load_state_dict(pkg['generator'])
    print("Generator loaded successfully from checkpoint!")

    return generator

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

torch.cuda.empty_cache()
print('start')

# Define the path to the checkpoint
checkpoint_path = '/home/anhnct1/Documents/MimiTrainer/Log/spt_base/MimiTrainer_00011600'

generator_config = MeomeoConfig.from_pretrained('/home/anhnct1/Documents/MimiTrainer/config/meomeo_cfg.json')
# Instantiate the generator model
meomeo = MeomeoModel(config=generator_config)  # Replace with your generator class

# Load the generator from the checkpoint
meomeo = load_generator_only(path=checkpoint_path, generator=meomeo)

# Set the generator to evaluation mode for testing
meomeo.eval()

meomeo.push_to_hub(
    repo_id='phongtintruong/meomeo-mhubert-vietbud-2111401-11600',  # Name of the repository
    use_temp_dir=True,                # Temporarily saves files before pushing
    commit_message="Initial commit",  # Optional commit message
)