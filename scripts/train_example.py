# from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse
from transformers import MimiModel, AutoFeatureExtractor
from trainer import MimiTrainer



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue to train from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    generator = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    discriminators = {'mpd':MultiPeriodDiscriminator(), 'msd':MultiScaleDiscriminator(), 'mstftd':MultiScaleSTFTDiscriminator(32)}

    trainer = MimiTrainer(generator=generator,
                          feature_extractor=feature_extractor,
                          discriminators=discriminators,
                          cfg=cfg)

    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()