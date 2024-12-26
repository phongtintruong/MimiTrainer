from pathlib import Path
import re
import os
import itertools

from beartype import beartype

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import get_dataloader, RawAudioDataset
from .optimizer import get_optimizer
from torch.utils import tensorboard
from .loss import *
import json
import time
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from transformers import AutoFeatureExtractor, PreTrainedModel, Wav2Vec2CTCTokenizer
from transformers import EncodecFeatureExtractor, AutoProcessor, Wav2Vec2Processor
import torchaudio
from datasets import load_dataset  # Import Hugging Face datasets
from typing import Union, Dict, Any


# helpers

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
            epochs,
            batch_size,
            train_audio_path,
            val_audio_path,
            generator: PreTrainedModel,
            generator_feature_extractor: EncodecFeatureExtractor,
            teacher: PreTrainedModel,
            teacher_feature_extractor,
            generator_sampling_rate,
            teacher_sampling_rate,
            max_length_s,
            discriminators: dict,
            cfg,
            accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        torch.manual_seed(cfg.get('seed'))
        split_batches = cfg.get("split_batches", True)
        self.log_steps = cfg.get('log_steps')
        self.stdout_steps = cfg.get('stdout_steps')
        self.save_model_steps = cfg.get('save_model_steps')
        results_folder = cfg.get('results_folder')
        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = cfg.get("num_ckpt_keep")
        # self.epochs = cfg.get("epochs")
        self.epochs = epochs
        self.num_warmup_steps = cfg.get("num_warmup_steps")
        # self.batch_size = cfg.get("batch_size")
        self.batch_size = batch_size
        self.sample_rate = 24000
        self.showpiece_num = cfg.get('showpiece_num', 8)
        project_name = 'MimiTrainer'

        if not self.results_folder.exists():
            self.results_folder.mkdir(parents=True, exist_ok=True)

        with open(f'{str(self.results_folder)}/config.json', 'w+') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        # tracker = AudioTensorBoardTracker(run_name=project_name, logging_dir=results_folder)
        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            # log_with=tracker,
            **accelerate_kwargs
        )

        if self.is_main:
            self.writer = tensorboard.SummaryWriter(os.path.join(results_folder, 'logs'))

        self.train_audio_path = train_audio_path
        self.val_audio_path = val_audio_path
        self.generator = generator
        self.generator_feature_extractor = generator_feature_extractor
        self.teacher = teacher
        self.teacher_feature_extractor = teacher_feature_extractor
        self.generator_sampling_rate = generator_sampling_rate
        self.teacher_sampling_rate = teacher_sampling_rate
        self.max_length_s = max_length_s
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
                {'n_fft': cfg.get('n_fft') // mult, 'num_mels': cfg.get('num_mels'), 'sample_rate': self.sample_rate,
                 'hop_size': cfg.get('hop_size') // mult, 'win_size': cfg.get('win_size') // mult,
                 'fmin': cfg.get('fmin'),
                 'fmax': cfg.get('fmax_for_loss')})
            mult = mult * 2
        self.mel_kwargs = {'n_fft': cfg.get('n_fft'), 'num_mels': cfg.get('num_mels'), 'sample_rate': self.sample_rate,
                           'hop_size': cfg.get('hop_size'), 'win_size': cfg.get('win_size'), 'fmin': cfg.get('fmin'),
                           'fmax': cfg.get('fmax')}

        # Check if the datasets are Hugging Face datasets or local directories
        if isinstance(train_audio_path, str) and os.path.isdir(train_audio_path):
            # Local dataset
            self.ds = RawAudioDataset(data_dir=train_audio_path, mode='train')
        else:
            # Hugging Face dataset
            self.ds = load_dataset("parquet", data_files=train_audio_path)['train']

        if isinstance(val_audio_path, str) and os.path.isdir(val_audio_path):
            # Local dataset
            self.valid_ds = RawAudioDataset(data_dir=val_audio_path, mode='val')
        else:
            # Hugging Face dataset
            self.valid_ds = load_dataset("parquet", data_files=val_audio_path)['val'].select(range(100))

        if self.is_main:
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with {len(self.valid_ds)} samples')

        assert len(self.ds) >= self.batch_size, 'dataset must have sufficient samples for training'
        assert len(
            self.valid_ds) >= self.batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader
        drop_last = cfg.get("drop_last", True)
        num_workers = cfg.get("num_workers")
        self.dl = get_dataloader(self.ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last,
                                 num_workers=num_workers, feature_extractor_student=generator_feature_extractor,
                                 feature_extractor_teacher=teacher_feature_extractor,
                                 teacher_sampling_rate=teacher_sampling_rate,
                                 student_sampling_rate=generator_sampling_rate, max_length_s=max_length_s)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                       num_workers=4, feature_extractor_student=generator_feature_extractor,
                                       feature_extractor_teacher=teacher_feature_extractor,
                                       teacher_sampling_rate=teacher_sampling_rate,
                                       student_sampling_rate=generator_sampling_rate, max_length_s=max_length_s)

        # lr
        self.lr = cfg.get("learning_rate")
        self.initial_lr = cfg.get("intial_learning_rate")

        # optimizer
        self.optim_g = get_optimizer(
            generator.parameters(),
            lr=cfg.get("learning_rate"),
            wd=cfg.get("wd"),
            betas=cfg.get("betas")
        )

        self.optim_d = get_optimizer(
            itertools.chain(*[i.parameters() for i in self.discriminators.values()]),
            lr=cfg.get("learning_rate"),
            wd=cfg.get("wd"),
            betas=cfg.get("betas")
        )

        # scheduler
        # num_train_steps = epochs * self.ds.__len__() // (batch_size * grad_accum_every)
        num_train_steps = self.epochs * self.ds.__len__() // batch_size
        self.scheduler_g = CosineAnnealingLR(self.optim_g, T_max=num_train_steps)
        self.scheduler_d = CosineAnnealingLR(self.optim_d, T_max=num_train_steps)

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

        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": self.num_warmup_steps, "learning_rate": self.lr,
               "initial_learning_rate": self.initial_lr, "epochs": self.epochs}
        self.accelerator.init_trackers("Mimi", config=hps)
        self.best_dev_mel_loss = float('inf')
        self.plot_gt_once = False

    def save(self, path, best_dev_mel_loss):
        if best_dev_mel_loss < self.best_dev_mel_loss:
            self.best_dev_mel_loss = best_dev_mel_loss
            torch.save(self.accelerator.get_state_dict(self.generator), f'{self.results_folder}/Mimi_best_dev.pt')
        ckpts = sorted(Path(path).parent.glob(f'MimiTrainer_*'))
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[:-self.num_ckpt_keep]]
        pkg = dict(
            generator=self.accelerator.get_state_dict(self.generator),
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_g=self.optim_g.state_dict(),
            optim_d=self.optim_d.state_dict(),
            scheduler_g=self.scheduler_g.state_dict(),
            scheduler_d=self.scheduler_d.state_dict(),
            best_dev_mel_loss=self.best_dev_mel_loss
        )
        torch.save(pkg, path)

    def load(self, path=None, restore_optimizer=True):
        if not exists(path):
            ckpts = sorted(self.results_folder.glob(f'MimiTrainer_*'))
            path = str(ckpts[-1])
        generator = self.accelerator.unwrap_model(self.generator)
        pkg = torch.load(path, map_location='cpu')
        generator.load_state_dict(pkg['generator'])
        discriminators = {k: self.accelerator.unwrap_model(v) for k, v in self.discriminators.items()}
        map(lambda kv: kv[1].load_state_dict(pkg['discriminators'][kv[0]]), discriminators.items())

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

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

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

        self.generator.train()
        map(lambda disc: disc.train(), self.discriminators.values())
        step_time_log = {}

        steps = int(self.steps.item())
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler_d.step()
            self.scheduler_g.step()
            lr = self.scheduler_d.get_last_lr()[0]

        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch:{epoch} start...')

            for batch in self.dl:

                tic = time.time()

                x, inputs_teacher = batch
                # x = inputs_student.squeeze(0)
                # x = x.to(self.device)
                # inputs_student = inputs_student.to(self.device)
                # inputs_teacher = inputs_teacher.to(self.device)
                # print('x')
                # print(x)
                # print(x.shape)
                # print('inputs_student')
                # print(inputs_student)
                # print(inputs_student.shape)
                # print('inputs_teacher')
                # print(inputs_teacher)
                # print(inputs_teacher.shape)
                with torch.no_grad():
                    outputs_teacher = self.teacher(inputs_teacher)
                semantic_feature = outputs_teacher.last_hidden_state
                # print('org teacher_semantic token')
                # print(semantic_feature.shape)
                semantic_feature = nn.functional.pad(
                    semantic_feature.transpose(1, 2),  # Transpose to [Batch, Seq Length, Channels]
                    pad=(4, 4),  # Symmetric padding
                    mode="reflect"
                )
                semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8,
                                                            stride=4).transpose(1, 2)
                # print('teacher semantic token')
                # print(semantic_feature)
                # print(semantic_feature.shape)
                model_outs = self.generator(x)
                x_hat, feature = model_outs.audio_values, model_outs.semantic_token
                # print('x_hat')
                # print(x_hat)
                # print(x_hat.shape)
                # print('feature')
                # print(feature)
                # print(feature.shape)
                # print(len(feature))
                # print('discretes')
                # print(discretes)
                # print(discretes.shape)

                # Discriminators
                self.optim_d.zero_grad()
                discriminator_outputs = list(map(lambda disc: disc(x, x_hat.detach()), self.discriminators.values()))
                loss_disc_all = sum(map(lambda x: discriminator_loss(*x[:2]), discriminator_outputs))

                self.accelerator.backward(loss_disc_all)
                self.optim_d.step()

                # Generator
                self.optim_g.zero_grad()
                discriminator_outputs = list(map(lambda disc: disc(x, x_hat), self.discriminators.values()))
                loss_recon = recon_loss(x, x_hat)
                loss_mel = sum(map(lambda mel_k: mel_k[0] * mel_loss(x, x_hat, **mel_k[1]),
                                   zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)))
                loss_feature = sum(map(lambda x: feature_loss(*x[2:]), discriminator_outputs))
                loss_adversarial = sum(map(lambda x: adversarial_loss(x[1]), discriminator_outputs))
                loss_distill = self.distill_loss(feature, semantic_feature)
                loss_generator_all = loss_feature + loss_adversarial + loss_mel + loss_recon * self.recon_loss_lambda + self.distill_loss_lambda * loss_distill
                self.accelerator.backward(loss_generator_all)
                # if exists(self.max_grad_norm):
                #     self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim_g.step()

                # Remove the detached tensors from the computational graph
                x = x.detach()
                x_hat = x_hat.detach()

                step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                # self.accelerator.wait_for_everyone()

                # log
                if self.is_main and not (steps % self.stdout_steps):
                    with torch.inference_mode():
                        mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                    self.print(
                        f"Epoch {epoch} -- Step {steps}: Gen Loss: {loss_generator_all.item():0.3f}; Mel Error:{mel_error:0.3f}; Distill Loss: {loss_distill.item():0.3f}; Time cost per step: {step_time_log['time_cost'] / self.stdout_steps:0.3f}s")
                    step_time_log = {}
                if self.is_main and not (steps % self.log_steps):
                    self.log({"train/discriminators loss": loss_disc_all.item(),
                              "train/generator loss": loss_generator_all.item(),
                              "train/feature loss": loss_feature.item(),
                              "train/adversarial loss": loss_adversarial.item(), "train/mel loss": loss_mel.item(),
                              "train/mel error": mel_error, "train/distillation loss": loss_distill.item(),
                              "train/learning_rate": lr}, step=steps)

                self.accelerator.wait_for_everyone()

                # validate and save model
                if self.is_main and not (steps % self.save_model_steps) and steps != 0:

                    self.print('Validation start ...')
                    # validate
                    total_mel_error = 0.0
                    total_distill_loss = 0.0
                    num = 0
                    self.generator.eval()
                    with torch.inference_mode():
                        for i, batch in tqdm(enumerate(self.valid_dl)):
                            x, inputs_teacher = batch
                            # x = inputs_student.squeeze(0)
                            # x = x.to(self.device)
                            # inputs_student = inputs_student.to(self.device)
                            # inputs_teacher = inputs_teacher.to(self.device)
                            # print('x')
                            # print(x)
                            # print(x.shape)
                            # print('inputs_student')
                            # print(inputs_student)
                            # print(inputs_student.shape)
                            # print('inputs_teacher')
                            # print(inputs_teacher)
                            # print(inputs_teacher.shape)
                            with torch.no_grad():
                                outputs_teacher = self.teacher(inputs_teacher)
                            semantic_feature = outputs_teacher.last_hidden_state
                            # print('org teacher_semantic token')
                            # print(semantic_feature.shape)
                            semantic_feature = nn.functional.pad(
                                semantic_feature.transpose(1, 2),  # Transpose to [Batch, Seq Length, Channels]
                                pad=(4, 4),  # Symmetric padding
                                mode="reflect"
                            )
                            semantic_feature = nn.functional.avg_pool1d(semantic_feature, kernel_size=8,
                                                                        stride=4).transpose(1, 2)

                            # print(x.shape)
                            # x = x.unsqueeze(1)
                            # x = x.squeeze().numpy()

                            # print('teacher semantic token')
                            # print(semantic_feature)
                            # print(semantic_feature.shape)
                            model_outs = self.generator(x)
                            x_hat, feature = model_outs.audio_values, model_outs.semantic_token
                            # print('x_hat')
                            # print(x_hat)
                            # print(x_hat.shape)
                            # print('feature')
                            # print(feature)
                            # print(feature.shape)
                            # print(len(feature))
                            # print('discretes')
                            # print(discretes)
                            # print(discretes.shape)

                            mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                            total_mel_error += mel_error
                            loss_distill = self.distill_loss(feature, semantic_feature).item()
                            total_distill_loss += loss_distill
                            num += x.size(0)
                            if i < self.showpiece_num:
                                if not self.plot_gt_once:
                                    self.log({f'groundtruth/x_{i}': x[0].cpu().detach()}, type='audio',
                                             sample_rate=self.sample_rate, step=steps)
                                    x_spec = mel_spectrogram(x.squeeze(1), **self.mel_kwargs)
                                    self.log({f'groundtruth/x_spec_{i}': plot_spectrogram(x_spec[0].cpu().numpy())},
                                             type='figure', step=steps)

                                self.log({f'generate/x_hat_{i}': x_hat[0].cpu().detach()}, type='audio',
                                         sample_rate=self.sample_rate, step=steps)
                                x_hat_spec = mel_spectrogram(x_hat.squeeze(1), **self.mel_kwargs)
                                self.log({f'generate/x_hat_spec_{i}': plot_spectrogram(x_hat_spec[0].cpu().numpy())},
                                         type='figure', step=steps)
                            # Remove the detached tensors from the computational graph
                            x = x.detach()
                            x_hat = x_hat.detach()
                        if not self.plot_gt_once:
                            self.plot_gt_once = True
                        self.print(
                            f'{steps}: dev mel error: {total_mel_error / num:0.3f}\tdev distill loss: {total_distill_loss / num:0.3f}')
                        self.log(
                            {'dev/mel error': total_mel_error / num, 'dev/distillation loss': total_distill_loss / num},
                            step=steps)

                    # save model
                    model_path = str(self.results_folder / f'MimiTrainer_{steps:08d}')
                    self.save(model_path, total_mel_error / num)
                    self.print(f'{steps}: saving model to {str(self.results_folder)}')
                    self.generator.train()
                    print('back to train')

                # Update lr
                self.steps += 1
                steps = int(self.steps.item())
                if steps < self.num_warmup_steps:
                    lr = self.warmup(steps)
                    for param_group in self.optim_g.param_groups:
                        param_group['lr'] = lr
                    for param_group in self.optim_d.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler_d.step()
                    self.scheduler_g.step()
                    lr = self.scheduler_g.get_last_lr()[0]

                # Explicitly delete to manage memory
                del x
                # del inputs_student
                del inputs_teacher
                del semantic_feature
                del model_outs
                # del discretes
                del x_hat
                del feature
                del discriminator_outputs
                del loss_disc_all
                del loss_recon
                del loss_mel
                del loss_feature
                del loss_adversarial
                del loss_distill
                del loss_generator_all

                torch.cuda.empty_cache()

        self.print('training complete')

    def continue_train(self):
        self.load()
        self.train()