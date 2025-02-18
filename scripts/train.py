import argparse
import json
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoFeatureExtractor
from mimitransformers import MeomeoModel, MeomeoConfig, SemanticTeacher
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
from mimitrainer.trainer import MimiTrainer

# Load environment variables
load_dotenv()

if __name__ == '__main__':
    # ✅ Add argument parser
    parser = argparse.ArgumentParser(description="Train mHuBERT with pre-trained discriminators.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config JSON file.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN_WRITE"), help="Hugging Face API token.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file.")
    args = parser.parse_args()

    # ✅ Use the provided config path
    config_file = Path(args.config_path)
    hf_token = args.hf_token
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found at {args.config_path}")

    # Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else config.get("checkpoint_path")

    # Login to Hugging Face Hub
    # hf_token = os.getenv("HF_TOKEN")
    print(hf_token)
    login(hf_token)

    # Load generator model & feature extractor
    generator_config = MeomeoConfig.from_pretrained(config_file)
    generator = MeomeoModel.from_pretrained("kyutai/mimi", config=generator_config)
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    print('meomeoloaded')

    # Load teacher model
    processor = AutoFeatureExtractor.from_pretrained(config["teacher_feature_extractor"])
    model = SemanticTeacher.from_pretrained(config["teacher_model_path"], config["teacher_model_type"])

    # Function to load discriminators from a checkpoint
    def load_discriminators(path, discriminators):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        pkg = torch.load(path, map_location="cpu")
        if "discriminators" not in pkg:
            raise KeyError("The checkpoint does not contain the discriminators state_dict.")
        for name, discriminator in discriminators.items():
            if name in pkg["discriminators"]:
                discriminator.load_state_dict(pkg["discriminators"][name])
                print(f"✅ Loaded discriminator '{name}' from checkpoint.")
            else:
                raise KeyError(f"❌ Missing state_dict for discriminator '{name}' in checkpoint.")
        return discriminators

    # Initialize discriminators
    discriminators = {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(config["msstft_disc_filters"]),
    }

    # Load pretrained discriminator weights
    pretrained_disc_path = config.get("pretrained_disc_path")
    if pretrained_disc_path:
        discriminators = load_discriminators(pretrained_disc_path, discriminators)

    # Create trainer
    trainer = MimiTrainer(
        generator=generator,
        generator_feature_extractor=feature_extractor,
        teacher=model,
        teacher_feature_extractor=processor,
        discriminators=discriminators,
        cfg=config,
        accelerate_kwargs={},
    )

    # Start or continue training
    # checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        trainer.continue_train(checkpoint_path)
    else:
        trainer.train()
