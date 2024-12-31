print('hello')
import torch
from transformers import MimiConfig
from mimitransformers import TrainingMimiProjectorModel, TrainingMimiProjectorConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == '__main__':
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    torch.cuda.empty_cache()
    print('start')

    def load_trained_model(generator_model, checkpoint_path, device='cuda'):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if the checkpoint contains the generator weights directly
        generator_model.load_state_dict(checkpoint)

        # Move model to the specified device
        generator_model = generator_model.to(device)

        return generator_model
    
    config = TrainingMimiProjectorConfig.from_pretrained("Log/spt_base/config.json")
    generator = TrainingMimiProjectorModel(config=config)
    checkpoint_path = 'Log/spt_base/Mimi_best_dev.pt'   
    generator = load_trained_model(generator, checkpoint_path, device='cuda')

    # Use the generator model
    generator.eval()  # Set the model to evaluation mode

    generator.push_to_hub(
        repo_id='phongtintruong/meomeo-mhubert-overfit-8-v0.3',  # Name of the repository
        use_temp_dir=True,                # Temporarily saves files before pushing
        commit_message="Initial commit",  # Optional commit message
    )

