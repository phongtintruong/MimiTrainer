# from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse
from mimitransformers import TrainingMimiModel, TrainingMimiProjectorModel
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
from transformers import AutoFeatureExtractor
from mimitrainer.trainer import MimiTrainer
from pathlib import Path
from transformers import AutoProcessor, WavLMModel
from huggingface_hub import login


if __name__ == '__main__':
    login("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")

    # Configuration
    CONFIG_PATH = "config/spt_base_cfg.json"  # Path to your config file

    # Load config from file
    config_file = Path(CONFIG_PATH)
    if config_file.is_file():
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    # Instantiate model and feature extractor
    generator = TrainingMimiProjectorModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")
    model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")

    discriminators = {
        'mpd': MultiPeriodDiscriminator(),
        'msd': MultiScaleDiscriminator(),
        'mstftd': MultiScaleSTFTDiscriminator(32)
    }

    train_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/train-00000-of-00105-be5f872f8be772f5.parquet"
    val_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"

    train_data_files = {"train": train_url}
    val_data_files = {'val': val_url}

    # Create trainer instance
    trainer = MimiTrainer(
        epochs=2,
        batch_size=2,
        train_audio_path=train_data_files,
        val_audio_path=val_data_files,
        generator=generator,
        generator_feature_extractor=feature_extractor,
        teacher=model,
        teacher_feature_extractor=processor,
        generator_sampling_rate=24000,
        teacher_sampling_rate=16000,
        max_length_s=3,
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