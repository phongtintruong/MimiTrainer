# from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse
from mimitransformers import TrainingMimiProjectorModel, TrainingMimiProjectorConfig
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
from transformers import AutoFeatureExtractor
from mimitrainer.trainer import MimiTrainer
from pathlib import Path
from transformers import AutoFeatureExtractor, HubertModel


if __name__ == '__main__':

    CONFIG_PATH = "config/spt_base_cfg.json"  # Path to your config file

    # Load config from file
    config_file = Path(CONFIG_PATH)
    if config_file.is_file():
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    # Instantiate model and feature extractor
    generator_config = TrainingMimiProjectorConfig.from_pretrained("config/spt_base_cfg.json")
    generator = TrainingMimiProjectorModel.from_pretrained("kyutai/mimi", config=generator_config)
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    processor = AutoFeatureExtractor.from_pretrained(config["teacher_feature_extractor"])
    model = HubertModel.from_pretrained(config["teacher_model_path"])

    discriminators = {
        'mpd': MultiPeriodDiscriminator(),
        'msd': MultiScaleDiscriminator(),
        'mstftd': MultiScaleSTFTDiscriminator(32)
    }

    # Create trainer instance
    trainer = MimiTrainer(
        generator=generator,
        generator_feature_extractor=feature_extractor,
        teacher=model,
        teacher_feature_extractor=processor,
        discriminators=discriminators,
        cfg=config,
        accelerate_kwargs={'mixed_precision': 'fp16'},
    )

    # Start training (or continue training)
    continue_training = False  # Set to True if you want to continue

    if continue_training:
        trainer.continue_train()
    else:
        trainer.train()