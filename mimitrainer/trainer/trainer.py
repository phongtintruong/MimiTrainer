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
from datasets import load_dataset  # Import Hugging Face datasets
import random


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
            teacher: PreTrainedModel,
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
        self.log_steps = cfg.get('log_steps')
        self.stdout_steps = cfg.get('stdout_steps')
        self.save_model_steps = cfg.get('save_model_steps')
        results_folder = cfg.get('results_folder')
        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = cfg.get("num_ckpt_keep")
        self.epochs = cfg.get("epochs")
        self.gradient_accumulation_steps = cfg.get("gradient_accumulation_steps")
        self.num_warmup_steps = cfg.get("num_warmup_steps")
        self.batch_size = cfg.get("batch_size")
        self.showpiece_num = cfg.get('showpiece_num', 8)
        self.sampling_rate = cfg.get('sampling_rate')
        self.max_length_s = cfg.get('max_length_s')
        self.teacher_sampling_rate = cfg.get('teacher_sampling_rate')
        self.train_audio_path = cfg.get('train_audio_path')
        self.val_audio_path = cfg.get('val_audio_path')
        self.train_files = cfg.get("train_files", None)
        self.val_files = cfg.get("val_files", None)
        self.stream_train_data = cfg.get("stream_train_data", True)
        self.stream_val_data = cfg.get("stream_val_data", False)
        self.train_est_len = cfg.get("train_est_len", None)
        self.val_est_len = cfg.get("val_est_len", None)
        self.ema_freq = cfg.get("ema_freq", 1)
        self.ema_generator = cfg.get("ema_generator", True)
        self.ema_discriminators = cfg.get("ema_discriminators", False)

        self.max_nq = cfg.get('max_nq', 8)
        self.quantization_rate = cfg.get('quantization_rate', 0.5)
        project_name = 'MimiTrainer'

        if not self.results_folder.exists():
            self.results_folder.mkdir(parents=True, exist_ok=True)

        with open(f'{str(self.results_folder)}/config.json', 'w+') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        # tracker = AudioTensorBoardTracker(run_name=project_name, logging_dir=results_folder)
        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
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
        distill_type = cfg.get('distill_type', 'd_axis')
        if distill_type == 't_axis':
            from functools import partial
            lambda_sim = cfg.get('lambda_sim', 1)
            self.distill_loss = partial(t_axis_distill_loss, lambda_sim=lambda_sim)
        else:
            self.distill_loss = d_axis_distill_loss

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
        self.lr = cfg.get("learning_rate")
        self.initial_lr = cfg.get("initial_learning_rate")

        # optimizer
        if self.ema_generator:
            self.optim_g, self.ema_g = get_optimizer_with_ema(
                self.generator,
                lr=cfg.get("learning_rate"),
                wd=cfg.get("wd"),
                betas=cfg.get("betas"),
                use_ema=True,
                ema_decay=cfg.get("ema_decay"),
                wd_module=cfg.get("wd_module"),
                wd_ndim=cfg.get("wd_ndim")
            )
            self.ema_g = self.ema_g['model_0'] # hard code will be changed later
        else:
            self.optim_g = get_optimizer_with_ema(
                self.generator,
                lr=cfg.get("learning_rate"),
                wd=cfg.get("wd"),
                betas=cfg.get("betas"),
                use_ema=False,
                ema_decay=cfg.get("ema_decay"),
                wd_module=cfg.get("wd_module"),
                wd_ndim=cfg.get("wd_ndim")
            )

        if self.ema_discriminators:
            self.optim_d, self.ema_ds = get_optimizer_with_ema(
                self.discriminators,
                lr=cfg.get("learning_rate"),
                wd=cfg.get("wd"),
                betas=cfg.get("betas"),
                use_ema=True,
                ema_decay=cfg.get("ema_decay"),
                wd_module=cfg.get("wd_module"),
                wd_ndim=cfg.get("wd_ndim")
            )
        else:
            self.optim_d = get_optimizer_with_ema(
                self.discriminators,
                lr=cfg.get("learning_rate"),
                wd=cfg.get("wd"),
                betas=cfg.get("betas"),
                use_ema=False,
                ema_decay=cfg.get("ema_decay"),
                wd_module=cfg.get("wd_module"),
                wd_ndim=cfg.get("wd_ndim")
            )

        # scheduler
        if self.stream_train_data:
            num_train_steps = self.epochs * self.train_est_len // (self.batch_size * self.gradient_accumulation_steps)
        else:
            num_train_steps = self.epochs * self.ds.__len__() // (self.batch_size * self.gradient_accumulation_steps)
        # self.scheduler_g = CosineAnnealingLR(self.optim_g, T_max=num_train_steps)
        # self.scheduler_d = CosineAnnealingLR(self.optim_d, T_max=num_train_steps)
        self.scheduler_g = get_cosine_schedule_with_warmup(
            self.optim_g,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_train_steps
        )

        self.scheduler_d = get_cosine_schedule_with_warmup(
            self.optim_d,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_train_steps
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

        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": self.num_warmup_steps, "learning_rate": self.lr,
               "initial_learning_rate": self.initial_lr, "epochs": self.epochs}
        self.accelerator.init_trackers("Meomeo", config=hps)
        self.best_dev_mel_loss = float('inf')
        self.plot_gt_once = False

    def save(self, path, best_dev_mel_loss):
        if best_dev_mel_loss <= self.best_dev_mel_loss:
            self.best_dev_mel_loss = best_dev_mel_loss
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
            best_dev_mel_loss=self.best_dev_mel_loss
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

            if 'best_dev_mel_loss' in pkg.keys():
                self.best_dev_mel_loss = pkg['best_dev_mel_loss']
                if self.is_main:
                    self.print(f'The best dev mel loss before is {self.best_dev_mel_loss}')

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

    def train(self):
        print(self.accelerator.gradient_accumulation_steps)
        self.generator.train()
        for disc in self.discriminators.values():
            disc.train()

        step_time_log = {}

        steps = int(self.steps.item())
        # if steps < self.num_warmup_steps:
        #     lr = self.warmup(steps)
        #     for param_group in self.optim_d.param_groups:
        #         param_group['lr'] = lr
        #     for param_group in self.optim_g.param_groups:
        #         param_group['lr'] = lr
        # else:
        #     self.scheduler_d.step()
        #     self.scheduler_g.step()
        #     lr = self.scheduler_d.get_last_lr()[0]

        lr = self.scheduler_g.get_last_lr()[0]

        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch {epoch} start...')

            for batch in self.dl:
                self.generator.train()
                tic = time.time()
                x, inputs_teacher = batch

                # Forward pass for teacher
                with torch.no_grad():
                    outputs_teacher = self.teacher(inputs_teacher)
                semantic_feature = nn.functional.pad(
                    outputs_teacher.last_hidden_state.transpose(1, 2),
                    pad=(4, 4),
                    mode="reflect"
                )
                semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8, stride=4).transpose(1, 2)

                # Generator forward pass
                # if int(self.steps.item()) % self.gradient_accumulation_steps == 0:
                #     do_quantize = random.choices([True, False], weights=[self.quantization_rate, 1 - self.quantization_rate], k=1)[0]
                #     nq = random.randint(2, self.max_nq+1) #hard code will be changed later
                #     print('nq', nq)
                #     print('do_quantize', do_quantize)

                do_quantize = random.choices([True, False], weights=[self.quantization_rate, 1 - self.quantization_rate], k=1)[0]
                nq = random.randint(1, self.max_nq+1)
                model_outs = self.generator(input_values=x, num_quantizers=nq, do_quantize=do_quantize)
                x_hat, feature = model_outs.audio_values, model_outs.semantic_tokens
                # print('x', x.shape)
                # print('x_hat', x_hat.shape)
                # print('feature', feature.shape)

                # Discriminator update
                with self.accelerator.accumulate(self.discriminators):
                    discriminator_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]
                    loss_disc_all = sum(discriminator_loss(*output[:2]) for output in discriminator_outputs)
                    self.accelerator.backward(loss_disc_all)
                    self.optim_d.step()
                    self.scheduler_d.step()
                    if self.ema_discriminators:
                        for name, ema_disc in self.ema_ds.items():
                            ema_disc.update_parameters(self.discriminators[name])
                    self.optim_d.zero_grad()

                # Generator update
                with self.accelerator.accumulate(self.generator):
                    discriminator_outputs = [disc(x, x_hat) for disc in self.discriminators.values()]
                    loss_recon = recon_loss(x, x_hat)
                    loss_mel = sum(
                        mel_lambda * mel_loss(x, x_hat, **mel_kwargs)
                        for mel_lambda, mel_kwargs in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
                    )
                    loss_feature = sum(feature_loss(*output[2:]) for output in discriminator_outputs)
                    loss_adversarial = sum(adversarial_loss(output[1]) for output in discriminator_outputs)
                    loss_distill = self.distill_loss(feature, semantic_feature)

                    loss_generator_all = (
                        loss_feature +
                        loss_adversarial +
                        loss_mel +
                        loss_recon * self.recon_loss_lambda +
                        loss_distill * self.distill_loss_lambda
                    )
                    self.accelerator.backward(loss_generator_all)
                    self.optim_g.step()
                    self.scheduler_g.step()
                    if self.ema_generator:
                        self.ema_g.update_parameters(self.generator)
                    self.optim_g.zero_grad()

                # Learning rate update
                self.steps += 1
                steps = int(self.steps.item())
                lr = self.scheduler_g.get_last_lr()[0]
                # if steps % self.gradient_accumulation_steps == 0:
                #     accumulated_steps = steps // self.gradient_accumulation_steps
                #     if accumulated_steps < self.num_warmup_steps:
                #         lr = self.warmup(accumulated_steps)
                #         for optim in [self.optim_d, self.optim_g]:
                #             for param_group in optim.param_groups:
                #                 param_group['lr'] = lr
                #     else:
                #         self.scheduler_d.step()
                #         self.scheduler_g.step()
                #         lr = self.scheduler_g.get_last_lr()[0]


                # Logging
                step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                if steps % self.gradient_accumulation_steps == 0:
                    accumulated_steps = steps // self.gradient_accumulation_steps - 1
                    if self.is_main and not (accumulated_steps % self.stdout_steps):
                        with torch.no_grad():
                            mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                        self.print(
                            f"Epoch {epoch} -- Step {accumulated_steps}: "
                            f"Gen Loss: {loss_generator_all.item():0.3f}; "
                            f"Mel Error: {mel_error:0.3f}; "
                            f"Distill Loss: {loss_distill.item():0.3f}; "
                            f"Time cost per step: {step_time_log['time_cost'] / self.stdout_steps:0.3f}s"
                        )
                        step_time_log = {}

                    if self.is_main and not (accumulated_steps % self.log_steps):
                        self.log({
                            "train/discriminators loss": loss_disc_all.item(),
                            "train/generator loss": loss_generator_all.item(),
                            "train/feature loss": loss_feature.item(),
                            "train/adversarial loss": loss_adversarial.item(),
                            "train/mel loss": loss_mel.item(),
                            "train/mel error": mel_error,
                            "train/distillation loss": loss_distill.item(),
                            "train/learning_rate": lr
                        }, step=accumulated_steps)

                self.accelerator.wait_for_everyone()

                # validation and save
                if steps % self.gradient_accumulation_steps == 0:
                    accumulated_steps = steps // self.gradient_accumulation_steps - 1
                    if self.is_main and not (accumulated_steps % self.save_model_steps) and accumulated_steps != 0: #??
                        self.print('Validation start ...')
                        total_mel_error = 0.0
                        total_distill_loss = 0.0
                        num = 0
                        self.generator.eval()
                        with torch.inference_mode():
                            for i, batch in tqdm(enumerate(self.valid_dl)):
                                # print('validating')
                                x, inputs_teacher = batch
                                # with torch.no_grad():
                                outputs_teacher = self.teacher(inputs_teacher)
                                # print(f"inputs_teacher device: {inputs_teacher.device}")
                                # print('teacher output')
                                # inputs_teacher = inputs_teacher.to(self.device)
                                # print(f"inputs_teacher device: {inputs_teacher.device}")
                                semantic_feature = nn.functional.pad(
                                    outputs_teacher.last_hidden_state.transpose(1, 2),
                                    pad=(4, 4),
                                    mode="reflect"
                                )
                                semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8, stride=4).transpose(1, 2)

                                model_outs = self.generator(input_values=x, num_quantizers=self.max_nq, do_quantize=True)
                                x_hat, feature = model_outs.audio_values, model_outs.semantic_tokens
                                # print('generator output')

                                mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                                distill_loss = self.distill_loss(feature, semantic_feature).item()

                                total_mel_error += mel_error
                                total_distill_loss += distill_loss
                                num += x.size(0)
                                if i < self.showpiece_num:
                                    if not self.plot_gt_once:
                                        self.log({f'groundtruth/x_{i}': x[0].cpu().detach()}, type='audio',
                                                sample_rate=self.sampling_rate, step=accumulated_steps)
                                        x_spec = mel_spectrogram(x.squeeze(1), **self.mel_kwargs)
                                        self.log({f'groundtruth/x_spec_{i}': plot_spectrogram(x_spec[0].cpu().numpy())},
                                                type='figure', step=accumulated_steps)

                                    self.log({f'generate/x_hat_{i}': x_hat[0].cpu().detach()}, type='audio',
                                            sample_rate=self.sampling_rate, step=accumulated_steps)
                                    x_hat_spec = mel_spectrogram(x_hat.squeeze(1), **self.mel_kwargs)
                                    self.log({f'generate/x_hat_spec_{i}': plot_spectrogram(x_hat_spec[0].cpu().numpy())},
                                            type='figure', step=accumulated_steps)
                                # # Remove the detached tensors from the computational graph
                                # x = x.detach()
                                # x_hat = x_hat.detach()
                            if not self.plot_gt_once:
                                self.plot_gt_once = True
                            self.print(
                                f'{accumulated_steps}: dev mel error: {total_mel_error / num:0.3f}\tdev distill loss: {total_distill_loss / num:0.3f}')
                            self.log(
                                {'dev/mel error': total_mel_error / num, 'dev/distillation loss': total_distill_loss / num},
                                step=accumulated_steps)
                            
                        # save model
                        model_path = str(self.results_folder / f'MimiTrainer_{accumulated_steps:08d}')
                        self.save(model_path, (total_mel_error / num) + (total_distill_loss / num) * 2)
                        self.print(f'{accumulated_steps}: saving model to {str(self.results_folder)}')
                        # self.generator.train()
                        print('back to train')

            # Save model at the end of the epoch
            if epoch == self.epochs - 1:
                model_path = str(self.results_folder / f'MimiTrainer_last')
                self.save(model_path, self.best_dev_mel_loss + 1)
                self.print(f'{epoch}: saving model to {str(self.results_folder)}')
                # self.generator.train()

        self.print('Training complete')


    def continue_train(self):
        self.load()
        self.train()