# from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse
from mimitransformers import MeomeoModel, MeomeoConfig
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
from transformers import AutoFeatureExtractor
from mimitrainer.trainer import MimiTrainer
from pathlib import Path
from transformers import AutoFeatureExtractor, HubertModel
from huggingface_hub import login
import os, torch


if __name__ == '__main__':
    login('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')

    CONFIG_PATH = "config/spt_base_cfg_pretrained_discriminators.json"  # Path to your config file

    # Load config from file
    config_file = Path(CONFIG_PATH)
    if config_file.is_file():
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    # Instantiate model and feature extractor
    generator_config = MeomeoConfig.from_pretrained("config/meomeo_cfg.json")
    generator = MeomeoModel.from_pretrained("kyutai/mimi", config=generator_config)
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    processor = AutoFeatureExtractor.from_pretrained(config["teacher_feature_extractor"])
    model = HubertModel.from_pretrained(config["teacher_model_path"])

    def load_discriminators(path, discriminators):
        # Check if the checkpoint file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        # Load the checkpoint
        pkg = torch.load(path, map_location='cpu')
        print(f"Checkpoint keys: {pkg.keys()}")  # Print checkpoint keys for debugging

        # Ensure the checkpoint contains discriminator state_dicts
        if 'discriminators' not in pkg:
            raise KeyError("The checkpoint does not contain the discriminators state_dict.")

        # Load each discriminator's state_dict
        for name, discriminator in discriminators.items():
            if name not in pkg['discriminators']:
                raise KeyError(f"The checkpoint does not contain the state_dict for discriminator '{name}'.")
            discriminator.load_state_dict(pkg['discriminators'][name])  # Load state_dict for each discriminator
            print(f"Discriminator '{name}' loaded successfully from checkpoint.")

        return discriminators

    discriminators = {
        # 'mpd': MultiPeriodDiscriminator(),
        # 'msd': MultiScaleDiscriminator(),
        'mstftd': MultiScaleSTFTDiscriminator(32)
    }

    pretrained_discriminators_path = '/kaggle/input/mimi_discriminator/other/2450/1/Pretrained_Discriminators_00002450'

    discriminators = load_discriminators(path=pretrained_discriminators_path, discriminators=discriminators)

    # Create trainer instance
    trainer = MimiTrainer(
        generator=generator,
        generator_feature_extractor=feature_extractor,
        teacher=model,
        teacher_feature_extractor=processor,
        discriminators=discriminators,
        cfg=config,
        accelerate_kwargs={},
    )

    # Start training (or continue training)
    continue_training = False  # Set to True if you want to continue

    if continue_training:
        trainer.continue_train()
    else:
        trainer.train()