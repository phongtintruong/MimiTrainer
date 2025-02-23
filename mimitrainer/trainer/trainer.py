from pathlib import Path
import re
import os
import itertools

from beartype import beartype

import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup

from .dataset import get_dataloader, RawAudioDataset
from .optimizer import get_optimizer_with_ema
from torch.utils import tensorboard
from .loss import *
import json
import time
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration, DistributedType
from transformers import PreTrainedModel, EncodecFeatureExtractor
from mimitransformers import SemanticTeacher
from datasets import load_dataset  # Import Hugging Face datasets


# helpers
def read_urls_from_file(filename):
    urls = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip the newline character and append to the list
            urls.append(line.strip())
    return urls


def exists(val):
    return val is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename."""
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


class MimiTrainer(nn.Module):
    @beartype
    def __init__(
            self,
            generator: PreTrainedModel,
            generator_feature_extractor: EncodecFeatureExtractor,
            teacher: SemanticTeacher,
            teacher_feature_extractor,
            discriminators: dict,
            cfg,
            accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        self.find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.find_unused_parameters)
        torch.manual_seed(cfg.get('seed'))
        split_batches = cfg.get("split_batches", True)
        self.disc_log_steps = cfg.get('disc_log_steps')
        self.gen_log_steps = cfg.get('gen_log_steps')
        # self.stdout_steps = cfg.get('stdout_steps')
        self.save_model_steps = cfg.get('save_model_steps')
        self.save_pretrained_discriminators_steps = cfg.get('save_pretrained_discriminators_steps')
        results_folder = cfg.get('results_folder')
        self.results_folder = Path(results_folder)
        pretrained_discriminators_folder = cfg.get('pretrained_discriminators_folder')
        self.pretrained_discriminators_folder = Path(pretrained_discriminators_folder)
        self.num_ckpt_keep = cfg.get("num_ckpt_keep")
        self.epochs = cfg.get("epochs")
        self.gen_gradient_accumulation_steps = cfg.get("gen_gradient_accumulation_steps")
        self.disc_gradient_accumulation_steps = cfg.get("disc_gradient_accumulation_steps")
        self.gen_num_warmup_steps = cfg.get("gen_num_warmup_steps")
        self.disc_num_warmup_steps = cfg.get("disc_num_warmup_steps")
        self.discriminators_warmup_epochs = cfg.get("discriminators_warmup_epochs")
        self.batch_size = cfg.get("batch_size")
        self.showpiece_num = cfg.get('showpiece_num', 8)
        self.sampling_rate = cfg.get('sampling_rate')
        self.max_length_s = cfg.get('max_length_s')
        self.teacher_sampling_rate = cfg.get('teacher_sampling_rate')
        self.teacher_semantic_token_type = cfg.get('teacher_semantic_token_type')
        self.train_audio_path = cfg.get('train_audio_path')
        self.val_audio_path = cfg.get('val_audio_path')
        self.train_files = cfg.get("train_files", None)
        self.val_files = cfg.get("val_files", None)
        self.stream_train_data = cfg.get("stream_train_data", True)
        self.stream_val_data = cfg.get("stream_val_data", False)
        self.tqdm_valid = cfg.get('tqdm_valid', False)
        self.train_est_len = cfg.get("train_est_len", None)
        self.val_est_len = cfg.get("val_est_len", None)
        self.ema_freq = cfg.get("ema_freq", 1)
        self.ema_generator = cfg.get("ema_generator", True)
        self.ema_discriminators = cfg.get("ema_discriminators", False)


        self.quantization_rate = cfg.get('quantization_rate', 0.5)
        project_name = 'MimiTrainer'

        if not self.results_folder.exists():
            self.results_folder.mkdir(parents=True, exist_ok=True)

        if not self.pretrained_discriminators_folder.exists():
            self.pretrained_discriminators_folder.mkdir(parents=True, exist_ok=True)

        with open(f'{str(self.results_folder)}/config.json', 'w+') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        # tracker = AudioTensorBoardTracker(run_name=project_name, logging_dir=results_folder)
        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            # mixed_precision="fp16",
            kwargs_handlers=[ddp_kwargs],
            # log_with=tracker,
            **accelerate_kwargs
        )

        if self.is_main:
            self.writer = tensorboard.SummaryWriter(os.path.join(results_folder, 'logs'))

        self.generator = generator
        self.generator_feature_extractor = generator_feature_extractor
        self.teacher = teacher
        self.teacher_feature_extractor = teacher_feature_extractor
        self.discriminators = discriminators
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher.eval()

        self.register_buffer('steps', torch.Tensor([0]))

        self.mel_loss_lambdas = cfg.get('mel_loss_lambdas')
        # self.commitment_loss_lambda = cfg.get('commitment_loss_lambda')
        self.recon_loss_lambda = cfg.get('recon_loss_lambda')
        self.distill_loss_lambda = cfg.get('distill_loss_lambda')
        self.feature_loss_lambda = cfg.get('feature_loss_lambda')
        self.adversarial_loss_lambda = cfg.get('adversarial_loss_lambda')
        self.semantic_commitment_loss_lambda = cfg.get('semantic_commitment_loss_lambda')
        self.acoustic_commitment_loss_lambda = cfg.get('acoustic_commitment_loss_lambda')
        distill_type = cfg.get('distill_type', 'd_axis_mimi')
        if distill_type == 't_axis':
            from functools import partial
            lambda_sim = cfg.get('lambda_sim', 1)
            self.distill_loss = partial(t_axis_distill_loss, lambda_sim=lambda_sim)
        elif distill_type == 'd_axis_speechtokenizer':
            self.distill_loss = d_axis_distill_loss_speechtokenizer
        else:
            self.distill_loss = d_axis_distill_loss_mimi

        if self.mel_loss_lambdas != None:
            self.mel_loss_kwargs_list = []
            mult = 1
            for i in range(len(self.mel_loss_lambdas)):
                self.mel_loss_kwargs_list.append(
                    {'n_fft': cfg.get('n_fft') // mult, 'num_mels': cfg.get('num_mels'), 'sample_rate': self.sampling_rate,
                    'hop_size': cfg.get('hop_size') // mult, 'win_size': cfg.get('win_size') // mult,
                    'fmin': cfg.get('fmin'),
                    'fmax': cfg.get('fmax_for_loss')})
                mult = mult * 2
            self.mel_kwargs = {'n_fft': cfg.get('n_fft'), 'num_mels': cfg.get('num_mels'), 'sample_rate': self.sampling_rate,
                            'hop_size': cfg.get('hop_size'), 'win_size': cfg.get('win_size'), 'fmin': cfg.get('fmin'),
                            'fmax': cfg.get('fmax')}


        if self.train_audio_path == 'parquet':
            train_urls = read_urls_from_file(self.train_files)
            train_data_files = {"train": train_urls}
            self.ds = load_dataset("parquet", data_files=train_data_files, streaming=self.stream_train_data)['train']
        elif os.path.isdir(self.train_audio_path):
            self.ds = RawAudioDataset(data_dir=self.train_audio_path, mode='train')
        else:
            self.ds = load_dataset(self.train_audio_path, split='train', streaming=self.stream_train_data)

        if self.val_audio_path == 'parquet':
            val_urls = read_urls_from_file(self.val_files)
            val_data_files = {"val": val_urls}
            self.valid_ds = load_dataset("parquet", data_files=val_data_files, streaming=self.stream_val_data)['val']
        elif os.path.isdir(self.val_audio_path):    
            self.valid_ds = RawAudioDataset(data_dir=self.val_audio_path, mode='val')
        else:
            self.valid_ds = load_dataset(self.val_audio_path, split='val', streaming=self.stream_val_data)

        if self.is_main:
            if self.stream_train_data:
                self.print(f'training with dataset:\n{self.ds}\nvaidating with:\n{self.valid_ds}')
            else:
                self.print(
                    f'training with dataset of {len(self.ds)} samples and validating with {len(self.valid_ds)} samples')

        if not self.stream_train_data:
            assert len(self.ds) >= self.batch_size, 'dataset must have sufficient samples for training'
        if not self.stream_val_data:
            assert len(
                self.valid_ds) >= self.batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader
        drop_last = cfg.get("drop_last", True)
        num_workers = cfg.get("num_workers")
        self.dl = get_dataloader(self.ds, batch_size=self.batch_size, shuffle=not self.stream_train_data, drop_last=drop_last,
                                 num_workers=num_workers, feature_extractor_student=generator_feature_extractor,
                                 feature_extractor_teacher=teacher_feature_extractor,
                                 teacher_sampling_rate=self.teacher_sampling_rate,
                                 student_sampling_rate=self.sampling_rate, max_length_s=self.max_length_s)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size=self.batch_size, shuffle=not self.stream_val_data, drop_last=False,
                                       num_workers=4, feature_extractor_student=generator_feature_extractor,
                                       feature_extractor_teacher=teacher_feature_extractor,
                                       teacher_sampling_rate=self.teacher_sampling_rate,
                                       student_sampling_rate=self.sampling_rate, max_length_s=self.max_length_s)

        # lr
        self.lr = cfg.get("generator_learning_rate")
        self.initial_lr = cfg.get("initial_generator_learning_rate")

        # optimizer
        self.optim_g, self.ema_g = get_optimizer_with_ema(
            self.generator,
            lr=cfg.get("generator_learning_rate"),
            wd=cfg.get("wd"),
            betas=cfg.get("betas"),
            use_ema=self.ema_generator,
            ema_decay=cfg.get("ema_decay"),
            wd_module=cfg.get("wd_module"),
            wd_ndim=cfg.get("wd_ndim")
        )
        self.ema_g = self.ema_g['model_0'] # hard code will be changed later

        self.optim_d, self.ema_ds = get_optimizer_with_ema(
            self.discriminators,
            lr=cfg.get("discriminators_learning_rate"),
            wd=cfg.get("wd"),
            betas=cfg.get("betas"),
            use_ema=self.ema_discriminators,
            ema_decay=cfg.get("ema_decay"),
            wd_module=cfg.get("wd_module"),
            wd_ndim=cfg.get("wd_ndim")
        )
        

        self.generator_steps_skip = cfg.get('generator_steps_skip', 0)
        self.generator_start_late_steps = cfg.get('generator_start_late_steps', 0)
        if self.discriminators_warmup_epochs >= self.epochs:
            assert self.discriminators_warmup_epochs < self.epochs, 'discriminators_warmup_epochs must be less than epochs'


        # scheduler
        if self.stream_train_data:
            num_gen_train_steps = (self.epochs - self.discriminators_warmup_epochs) * ((self.train_est_len // self.batch_size) // self.gen_gradient_accumulation_steps) // (1 + self.generator_steps_skip) - self.generator_start_late_steps
            num_disc_train_steps = self.epochs * ((self.train_est_len // self.batch_size) // self.disc_gradient_accumulation_steps)
        else:
            # num_train_steps = self.epochs * self.ds.__len__() // (self.gradient_accumulation_steps * self.batch_size)
            num_gen_train_steps = (self.epochs - self.discriminators_warmup_epochs) * ((len(self.ds) // self.batch_size) // self.gen_gradient_accumulation_steps) // (1 + self.generator_steps_skip) - self.generator_start_late_steps
            num_disc_train_steps = self.epochs * ((len(self.ds) // self.batch_size) // self.disc_gradient_accumulation_steps)
        # self.scheduler_g = CosineAnnealingLR(self.optim_g, T_max=num_train_steps)
        # self.scheduler_d = CosineAnnealingLR(self.optim_d, T_max=num_train_steps)
        self.scheduler_g = get_cosine_schedule_with_warmup(
            self.optim_g,
            num_warmup_steps=int(self.gen_num_warmup_steps*num_gen_train_steps),
            num_training_steps=num_gen_train_steps  
        )

        self.scheduler_d = get_cosine_schedule_with_warmup(
            self.optim_d,
            num_warmup_steps=int(self.disc_num_warmup_steps*num_disc_train_steps),
            num_training_steps=num_disc_train_steps
        )

        # prepare with accelerator

        (
            self.generator,
            self.teacher,
            # self.feature_extractor,
            self.optim_g,
            self.optim_d,
            self.scheduler_g,
            self.scheduler_d,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.generator,
            self.teacher,
            # self.feature_extractor,
            self.optim_g,
            self.optim_d,
            self.scheduler_g,
            self.scheduler_d,
            self.dl,
            self.valid_dl
        )
        self.discriminators = {k: self.accelerator.prepare(v) for k, v in self.discriminators.items()}
        if self.ema_generator:
            self.ema_g = self.accelerator.prepare(self.ema_g)
        if self.ema_discriminators:
            self.ema_ds = {k: self.accelerator.prepare(v) for k, v in self.ema_ds.items()}

        hps = {"num_train_steps": num_gen_train_steps, "num_warmup_steps": self.gen_num_warmup_steps, "learning_rate": self.lr,
               "initial_learning_rate": self.initial_lr, "epochs": self.epochs}
        self.accelerator.init_trackers("Meomeo", config=hps)
        self.best_dev_loss = float('inf')
        self.plot_gt_once = False

    def save(self, path, best_dev_loss):
        if best_dev_loss <= self.best_dev_loss:
            self.best_dev_loss = best_dev_loss
            torch.save(self.accelerator.get_state_dict(self.generator), f'{self.results_folder}/Mimi_best_dev.pt')

        ckpts = sorted(Path(path).parent.glob(f'MimiTrainer_*'))
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[:-self.num_ckpt_keep]]

        # Prepare dictionary for saving all the necessary models, optimizers, and EMA states
        pkg = dict(
            generator=self.accelerator.get_state_dict(self.generator),
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_g=self.optim_g.state_dict(),
            optim_d=self.optim_d.state_dict(),
            scheduler_g=self.scheduler_g.state_dict(),
            scheduler_d=self.scheduler_d.state_dict(),
            best_dev_loss=self.best_dev_loss
        )

        # Include EMA models in the saved dictionary if they exist
        if self.ema_generator:
            pkg['ema_generator'] = self.accelerator.get_state_dict(self.ema_g)

        if self.ema_discriminators:
            pkg['ema_discriminators'] = {k: self.accelerator.get_state_dict(v) for k, v in self.ema_ds.items()}

        torch.save(pkg, path)


    def load(self, path=None, restore_optimizer=True):
        if not exists(path):
            ckpts = sorted(self.results_folder.glob(f'MimiTrainer_*'))
            path = str(ckpts[-1])

        generator = self.accelerator.unwrap_model(self.generator)
        pkg = torch.load(path, map_location='cpu')

        # Load the main generator and discriminators
        generator.load_state_dict(pkg['generator'])
        discriminators = {k: self.accelerator.unwrap_model(v) for k, v in self.discriminators.items()}
        map(lambda kv: kv[1].load_state_dict(pkg['discriminators'][kv[0]]), discriminators.items())

        # If optimizer and scheduler need to be restored
        if restore_optimizer:
            self.optim_d.load_state_dict(pkg['optim_d'])
            self.scheduler_d.load_state_dict(pkg['scheduler_d'])
            self.optim_g.load_state_dict(pkg['optim_g'])
            self.scheduler_g.load_state_dict(pkg['scheduler_g'])

            if 'best_dev_loss' in pkg.keys():
                self.best_dev_loss = pkg['best_dev_loss']
                if self.is_main:
                    self.print(f'The best dev loss before is {self.best_dev_loss}')

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

        # Load EMA models if they exist in the checkpoint
        if 'ema_generator' in pkg and self.ema_generator:
            self.ema_g.load_state_dict(pkg['ema_generator'])

        if 'ema_discriminators' in pkg and self.ema_discriminators:
            for k, v in self.ema_ds.items():
                v.load_state_dict(pkg['ema_discriminators'][k])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # def warmup(self, step):
    #     if step < self.num_warmup_steps:
    #         return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
    #     else:
    #         return self.lr

    def log(self, values: dict, step, type=None, **kwargs):
        if type == 'figure':
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == 'audio':
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)

    def get_teacher_semantic_token_by_last(self, inputs):
        with torch.no_grad():
            outputs_teacher = self.teacher(inputs).last_hidden_state
        semantic_feature = nn.functional.pad(
            outputs_teacher.transpose(1, 2),
            pad=(4, 4),
            mode="reflect"
        )
        semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8, stride=4).transpose(1, 2)
        return semantic_feature

    def get_teacher_semantic_token_by_mean(self, inputs):
        with torch.no_grad():
            outputs_teacher = self.teacher(inputs, output_hidden_states=True).hidden_states
        mean_hidden_states = torch.mean(torch.stack(outputs_teacher), dim=0)
        semantic_feature = nn.functional.pad(
            mean_hidden_states.transpose(1, 2),
            pad=(4, 4),
            mode="reflect"
        )
        semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8, stride=4).transpose(1, 2)
        return semantic_feature

    def get_all_generator_loss(self, x, x_hat, feature, semantic_feature, discriminator_outputs, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss, gradient_accumulation_steps):
        loss_recon = recon_loss(x, x_hat) / gradient_accumulation_steps
        loss_mel = sum(
            mel_lambda * mel_loss(x, x_hat, **mel_kwargs)
            for mel_lambda, mel_kwargs in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
        ) / gradient_accumulation_steps
        avg_mel_loss += loss_mel.item()

        loss_feature = sum(feature_loss(*output[2:]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_adversarial = sum(adversarial_loss(output[1]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_distill = self.distill_loss(feature, semantic_feature) / gradient_accumulation_steps

        loss_generator_all = (
            loss_feature * self.feature_loss_lambda +
            loss_adversarial * self.adversarial_loss_lambda +
            loss_distill * self.distill_loss_lambda +
            loss_recon * self.recon_loss_lambda +
            loss_mel
        )
        avg_generator_loss += loss_generator_all.item()
        avg_distill_loss += loss_distill.item()
        avg_feature_loss += loss_feature.item()
        avg_adversarial_loss += loss_adversarial.item()
        avg_recon_loss += loss_recon.item()
        return loss_generator_all, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss
    
    def get_generator_loss_without_mel(self, x, x_hat, feature, semantic_feature, discriminator_outputs, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss, gradient_accumulation_steps):
        loss_recon = recon_loss(x, x_hat) / gradient_accumulation_steps
        loss_feature = sum(feature_loss(*output[2:]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_adversarial = sum(adversarial_loss(output[1]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_distill = self.distill_loss(feature, semantic_feature) / gradient_accumulation_steps

        loss_generator_all = (
            loss_feature * self.feature_loss_lambda +
            loss_adversarial * self.adversarial_loss_lambda +
            loss_distill * self.distill_loss_lambda +
            loss_recon * self.recon_loss_lambda
        )
        avg_generator_loss += loss_generator_all.item()
        avg_distill_loss += loss_distill.item()
        avg_feature_loss += loss_feature.item()
        avg_adversarial_loss += loss_adversarial.item()
        avg_recon_loss += loss_recon.item()
        return loss_generator_all, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, 0
    
    def get_generator_loss_without_recon(self, x, x_hat, feature, semantic_feature, discriminator_outputs, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss, gradient_accumulation_steps):
        loss_mel = sum(
            mel_lambda * mel_loss(x, x_hat, **mel_kwargs)
            for mel_lambda, mel_kwargs in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
        ) / gradient_accumulation_steps
        avg_mel_loss += loss_mel.item()

        loss_feature = sum(feature_loss(*output[2:]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_adversarial = sum(adversarial_loss(output[1]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_distill = self.distill_loss(feature, semantic_feature) / gradient_accumulation_steps

        loss_generator_all = (
            loss_feature * self.feature_loss_lambda +
            loss_adversarial * self.adversarial_loss_lambda +
            loss_distill * self.distill_loss_lambda +
            loss_mel
        )
        avg_generator_loss += loss_generator_all.item()
        avg_distill_loss += loss_distill.item()
        avg_feature_loss += loss_feature.item()
        avg_adversarial_loss += loss_adversarial.item()
        avg_mel_loss += loss_mel.item()
        return loss_generator_all, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, 0, avg_mel_loss
    
    def get_generator_loss_without_mel_and_recon(self, x, x_hat, feature, semantic_feature, discriminator_outputs, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss, gradient_accumulation_steps):
        loss_feature = sum(feature_loss(*output[2:]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_adversarial = sum(adversarial_loss(output[1]) for output in discriminator_outputs) / gradient_accumulation_steps
        loss_distill = self.distill_loss(feature, semantic_feature) / gradient_accumulation_steps
        loss_generator_all = (
            loss_feature * self.feature_loss_lambda +
            loss_adversarial * self.adversarial_loss_lambda +
            loss_distill * self.distill_loss_lambda
        )
        avg_generator_loss += loss_generator_all.item()
        avg_distill_loss += loss_distill.item()
        avg_feature_loss += loss_feature.item()
        avg_adversarial_loss += loss_adversarial.item()
        return loss_generator_all, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, 0, 0
    
    def update_model(self, model, optim, scheduler, ema=None):
        optim.step()
        # print(f'epoch {epoch}, step {steps}')
        scheduler.step()
        optim.zero_grad()
        if ema:
            if type(ema) == dict:
                for name, ema_model in ema.items():
                    with torch.no_grad():
                        ema_model.update_parameters(model[name])
            else:
                with torch.no_grad():
                    ema.update_parameters(model)

    def log_loss(self, accelerator, losses, epoch, step, model_type, model_step, log_step):
        print(f"Epoch {epoch} -- Step {step} -- {model_type} steps {model_step}:", end=" ")
        for name, loss in losses.items():
            print(f"{name} Loss: {accelerator.gather(loss / log_step)}", end="; ")
            losses[name] = 0.
        print()


    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.generator.train()
        for disc in self.discriminators.values():
            disc.train()

        if self.teacher_semantic_token_type == 'last':
            get_teacher_semantic_token = self.get_teacher_semantic_token_by_last
        elif self.teacher_semantic_token_type == 'mean':
            get_teacher_semantic_token = self.get_teacher_semantic_token_by_mean

        if self.mel_loss_lambdas == None and self.recon_loss_lambda == None:
            get_generator_loss = self.get_generator_loss_without_mel_and_recon

        if self.mel_loss_lambdas == None and self.recon_loss_lambda != None:
            get_generator_loss = self.get_generator_loss_without_mel

        if self.mel_loss_lambdas != None and self.recon_loss_lambda == None:
            get_generator_loss = self.get_generator_loss_without_recon

        if self.mel_loss_lambdas != None and self.recon_loss_lambda != None:
            get_generator_loss = self.get_all_generator_loss

        step_time_log = {}

        steps = int(self.steps.item())

        lr = self.scheduler_g.get_last_lr()[0]

        avg_generator_loss = 0.
        avg_distill_loss = 0.

        avg_disc_loss = 0.
        avg_feature_loss = 0.
        avg_adversarial_loss = 0.
        avg_mel_loss = 0.
        avg_recon_loss = 0.

        avg_semantic_commit_loss = 0.
        avg_acoustic_commit_loss = 0.

        discriminators_steps = 0
        generator_steps = 0

        pretrain_disc_steps = 0
        batch_gen_steps = 0

        if self.stream_train_data:
            drop_last_point = (self.train_est_len // self.batch_size) // self.gen_gradient_accumulation_steps #todo: hard code
        else:
            drop_last_point = (len(self.ds) // self.batch_size) // self.gen_gradient_accumulation_steps

        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch {epoch} start...')

            for batch in self.dl:
                # print('step', steps)
                tic = time.time()
                x, inputs_teacher = batch

                semantic_feature = get_teacher_semantic_token(inputs_teacher)

                if epoch >= self.discriminators_warmup_epochs and (steps + 1) / self.gen_gradient_accumulation_steps > self.generator_start_late_steps:
                    for param in self.generator.parameters():
                            param.requires_grad = True
                    self.generator.train()

                    # for disc in self.discriminators.values():
                    #     for param in disc.parameters():
                    #         param.requires_grad = True
                    #     disc.train()

                    model_outs = self.generator(input_values=x)
                    x_hat, feature, semantic_commitment_loss, acoustic_commitment_loss = model_outs.audio_values, model_outs.semantic_tokens, model_outs.semantic_commitment_loss, model_outs.acoustic_commitment_loss
                    semantic_commitment_loss = semantic_commitment_loss / self.gen_gradient_accumulation_steps
                    acoustic_commitment_loss = acoustic_commitment_loss / self.gen_gradient_accumulation_steps
                    # x_hat.retain_grad()
                    # x_hat_detached = x_hat.detach().clone()
                    if torch.isnan(feature).any():
                        print("NaN detected in feature (student embedding)")
                    if torch.isnan(semantic_feature).any():
                        print("NaN detected in target_feature (teacher embedding)")
        
                    detach_discriminator_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]
                    
                    for disc in self.discriminators.values():
                        for param in disc.parameters():
                            param.requires_grad = False
                        disc.eval()
                    discriminator_outputs = [disc(x, x_hat) for disc in self.discriminators.values()]
                    for disc in self.discriminators.values():
                        for param in disc.parameters():
                            param.requires_grad = True
                        disc.train()

                    # discriminator_outputs = [tuple(tensor.detach() if isinstance(tensor, torch.Tensor) else tensor for tensor in disc(x, x_hat)) for disc in self.discriminators.values()]

# 
                    # x_hat_detached = x_hat.detach().clone()
                    # x_hat_clone = x_hat.clone()

                    # print("x_hat requires_grad:", x_hat.requires_grad)  # Should be True

                    loss_generator_all, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss = get_generator_loss(x, x_hat, feature, semantic_feature, discriminator_outputs, avg_generator_loss, avg_distill_loss, avg_feature_loss, avg_adversarial_loss, avg_recon_loss, avg_mel_loss, self.gen_gradient_accumulation_steps)
                    # print("Loss requires_grad:", loss_generator_all.requires_grad)  # Should be True
                    # self.accelerator.backward(loss_generator_all)
                    avg_semantic_commit_loss += semantic_commitment_loss
                    avg_acoustic_commit_loss += acoustic_commitment_loss
                    loss_generator_all = loss_generator_all + self.semantic_commitment_loss_lambda * semantic_commitment_loss + self.acoustic_commitment_loss_lambda * acoustic_commitment_loss
                    loss_generator_all.backward()
                    # if x_hat.grad is not None:
                    #     print('x_hat has grad')
                    # for name, disc in self.discriminators.items():
                    #     for param in disc.parameters():
                    #         if param.grad is not None:
                    #             print(f"{name} discriminator parameter {param.shape} has gradients!")
                    #             exit()
                    if (steps + 1) % self.gen_gradient_accumulation_steps == 0 and self.is_main:
                        batch_gen_steps += 1
                        generator_steps += 1
                        if not (generator_steps % self.ema_freq):
                            self.update_model(self.generator, self.optim_g, self.scheduler_g, self.ema_g)
                        else:
                            self.update_model(self.generator, self.optim_g, self.scheduler_g)
                            
                        if not (generator_steps % self.gen_log_steps):
                            loss_dict = {'Generator': avg_generator_loss, 'Distillation': avg_distill_loss, 'Feature': avg_feature_loss, 'Adversarial': avg_adversarial_loss, 'Reconstruction': avg_recon_loss, 'Mel': avg_mel_loss, 'Semantic Commit': avg_semantic_commit_loss, 'Acoustic Commit': avg_acoustic_commit_loss}
                            self.log({
                                "train/generator loss": self.accelerator.gather(avg_generator_loss / (self.gen_log_steps)),
                                "train/feature loss": self.accelerator.gather(avg_feature_loss / (self.gen_log_steps)),
                                "train/adversarial loss": self.accelerator.gather(avg_adversarial_loss / (self.gen_log_steps)),
                                "train/distillation loss": self.accelerator.gather(avg_distill_loss / (self.gen_log_steps)),
                                "train/learning_rate": lr
                            }, step=generator_steps)
                            self.log_loss(self.accelerator, loss_dict, epoch, steps, 'Generator', generator_steps, self.gen_log_steps)
                            avg_generator_loss = loss_dict['Generator']
                            avg_distill_loss = loss_dict['Distillation']
                            avg_feature_loss = loss_dict['Feature']
                            avg_adversarial_loss = loss_dict['Adversarial']
                            avg_recon_loss = loss_dict['Reconstruction']
                            avg_mel_loss = loss_dict['Mel']
                            avg_semantic_commit_loss = loss_dict['Semantic Commit']
                            avg_acoustic_commit_loss = loss_dict['Acoustic Commit']

                            self.print(
                                f"learning rate: {lr}; "
                                f"Time cost per step: {step_time_log['time_cost'] / self.gen_log_steps:0.3f}s"
                            )
                            step_time_log = {}

                        if not (generator_steps % self.save_model_steps):
                            self.print("Validation start ...")
                            total_distill_loss = 0.0
                            total_recon_loss = 0.0
                            total_disc_loss = 0.0
                            num = 0
                            self.generator.eval()
                            for disc in self.discriminators.values():
                                for param in disc.parameters():
                                    param.requires_grad = False
                                disc.eval()
                            with torch.no_grad():
                                iterator = tqdm(enumerate(self.valid_dl)) if self.tqdm_valid else enumerate(self.valid_dl)
                                for i, batch in iterator:
                                    # print('validating')
                                    x, inputs_teacher = batch
                                    # with torch.no_grad():
                                    semantic_feature = get_teacher_semantic_token(inputs_teacher)

                                    model_outs = self.generator(input_values=x, semantic_nq=1, acoustic_nq=7)
                                    x_hat, feature = model_outs.audio_values, model_outs.semantic_tokens
                                    # print('generator output')

                                    discriminator_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]
                                    loss_disc_all = sum(discriminator_loss(*output[:2]) for output in discriminator_outputs) 

                                    distill_loss = self.distill_loss(feature, semantic_feature).item()
                                    loss_recon = recon_loss(x, x_hat).item()

                                    total_distill_loss += distill_loss
                                    total_recon_loss += loss_recon
                                    total_disc_loss += loss_disc_all
                                    num += 1
                                    
                                self.print(
                                    f'{generator_steps}: dev recon loss: {total_recon_loss / num}\tdev disc loss: {total_disc_loss / num}\tdev distill loss: {total_distill_loss / num}')
                                self.log(
                                    {'dev/recon loss': total_recon_loss / num, 'dev/distillation loss': total_distill_loss / num},
                                    step=generator_steps)

                                # save model
                                model_path = str(self.results_folder / f'MimiTrainer_{generator_steps:08d}')
                                self.save(model_path, (total_distill_loss / num) + (total_recon_loss / num) * 2)
                                self.print(f'{generator_steps}: saving model to {str(self.results_folder)}')
                                # self.generator.train()
                                print('back to train')

                        # if batch_gen_steps >= drop_last_point:
                        #     self.steps += 1
                        #     steps = int(self.steps.item())
                        #     # Learning rate update
                        #     lr = self.scheduler_g.get_last_lr()[0]

                        #     step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                        #     batch_gen_steps = 0
                        #     break
                    
                    for disc in self.discriminators.values():
                        for param in disc.parameters():
                            param.requires_grad = True
                        disc.train()
                    loss_disc_all = sum(discriminator_loss(*output[:2]) for output in detach_discriminator_outputs) / self.disc_gradient_accumulation_steps
                    avg_disc_loss += loss_disc_all.item()
                    loss_disc_all.backward()
                    
                    if (steps + 1) % self.disc_gradient_accumulation_steps == 0 and self.is_main:
                        discriminators_steps += 1
                        if not (discriminators_steps % self.ema_freq):
                            self.update_model(self.discriminators, self.optim_d, self.scheduler_d, self.ema_ds)
                        else:
                            self.update_model(self.discriminators, self.optim_d, self.scheduler_d)

                        if not (discriminators_steps % self.disc_log_steps):
                            # print('Epoch', epoch, 'Step', steps, 'Discriminators_steps', discriminators_steps)
                            loss_dict = {'Discriminator': avg_disc_loss}
                            self.log_loss(self.accelerator, loss_dict, epoch, steps, 'Discriminator', discriminators_steps + pretrain_disc_steps, self.disc_log_steps)
                            avg_disc_loss = loss_dict['Discriminator']

                        
                        if batch_gen_steps >= drop_last_point:
                            self.steps += 1
                            steps = int(self.steps.item())
                            # Learning rate update
                            lr = self.scheduler_g.get_last_lr()[0]

                            step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                            batch_gen_steps = 0
                            break

                else: #todo: verify
                    for disc in self.discriminators.values():
                        for param in disc.parameters():
                            param.requires_grad = True
                        disc.train()
                    for param in self.generator.parameters():
                        param.requires_grad = False
                    self.generator.eval()

                    model_outs = self.generator(input_values=x)
                    x_hat, feature, semantic_commitment_loss, acoustic_commitment_loss = model_outs.audio_values, model_outs.semantic_tokens, model_outs.semantic_commitment_loss, model_outs.acoustic_commitment_loss
                    
                    if torch.isnan(feature).any():
                        print("NaN detected in feature (student embedding)")
                    if torch.isnan(semantic_feature).any():
                        print("NaN detected in target_feature (teacher embedding)")
        
                    detach_discriminator_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]

                    loss_disc_all = sum(discriminator_loss(*output[:2]) for output in detach_discriminator_outputs) / self.disc_gradient_accumulation_steps
                    avg_disc_loss += loss_disc_all.item()
                    loss_disc_all.backward()
                    # if x_hat.grad is not None:
                    #     print('x_hat has grad')
                    # found = False
                    # for name, disc in self.discriminators.items():
                    #     for param in disc.parameters():
                    #         if param.grad is not None:
                    #             print(f"{name} discriminator parameter {param.shape} has gradients!")
                    # if x_hat.grad is not None:
                    #     print('x_hat has grad')
                    # found = False
                    # for name, disc in self.discriminators.items():
                    #     for param in disc.parameters():
                    #         if param.grad is not None:
                    #             print(f"{name} discriminator parameter {param.shape} has gradients!")
                    #             found = True
                    #             break
                    #     if found:
                    #         break
                            # print(avg_disc_loss)
                    #             found = True
                    #             break
                    #     if found:
                    #         break
                    if (steps + 1) % self.disc_gradient_accumulation_steps == 0 and self.is_main:
                        pretrain_disc_steps += 1
                        if not (pretrain_disc_steps % self.ema_freq):
                            self.update_model(self.discriminators, self.optim_d, self.scheduler_d, self.ema_ds)
                        else:
                            self.update_model(self.discriminators, self.optim_d, self.scheduler_d)

                        if not (pretrain_disc_steps % self.disc_log_steps):
                            # print('Epoch', epoch, 'Step', steps, 'Discriminators_steps', discriminators_steps)
                            loss_dict = {'Discriminator': avg_disc_loss}
                            self.log_loss(self.accelerator, loss_dict, epoch, steps, 'Discriminator', pretrain_disc_steps, self.disc_log_steps)
                            avg_disc_loss = loss_dict['Discriminator']

                        if not (pretrain_disc_steps % self.save_pretrained_discriminators_steps):
                            # save model
                            model_path = str(
                                self.pretrained_discriminators_folder / f'Pretrained_Discriminators_{pretrain_disc_steps:08d}')
                            self.save(model_path, 99999)
                            self.print(
                                f'{pretrain_disc_steps}: saving model to {str(self.pretrained_discriminators_folder)}')
                        # elif discriminators_steps <= self.generator_start_late_steps:
                        #     loss_dict = {'Discriminator': avg_disc_loss}
                        #     self.log_loss(self.accelerator, loss_dict, epoch, steps, 'Discriminator', discriminators_steps, 1)
                        #     avg_disc_loss = loss_dict['Discriminator']

                        
                        if batch_gen_steps >= drop_last_point:
                            self.steps += 1
                            steps = int(self.steps.item())
                            # Learning rate update
                            lr = self.scheduler_g.get_last_lr()[0]

                            step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                            batch_gen_steps = 0
                            break

                self.steps += 1
                steps = int(self.steps.item())
                # Learning rate update
                lr = self.scheduler_g.get_last_lr()[0]
                step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})

            # Save model at the end of the epoch
            if epoch == self.epochs - 1:
                model_path = str(self.results_folder / f'MimiTrainer_last')
                self.save(model_path, self.best_dev_loss + 1)
                self.print(f'{epoch}: saving model to {str(self.results_folder)}')
                # self.generator.train()

        self.print('Training complete')


    def continue_train(self, checkpoint_path):
        self.load(path=checkpoint_path)
        self.train()