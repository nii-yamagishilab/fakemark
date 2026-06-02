"""Unified training entrypoint for SpeechBrain watermarking experiments.

This script supports both of the previously separate training flows:

- the SEANet encoder/decoder/conditioner pipeline from ``config/fakemark_a.yaml``
- the timbre encoder pipeline from ``config/fakemark_t.yaml``

The active watermarking backend is inferred from the configured modules by
default, but can also be set explicitly with ``watermark_backend`` in the
hyperparameter YAML.
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")
import textwrap
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import soundfile as sf
import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.data_utils import scalarize
from speechbrain.utils.logger import get_logger
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

from dataio.dataio_stc_merge import dataio_prepare as _dataio_prepare
from models.loss import freq_mag_loss
from perturbations.audio_effects import batchwise_random_perturbation
from util.utils import (
    StepCounter,
    get_watermark_eval,
    load_step_counter,
    save_step_counter,
    set_random_seed,
)

logger = get_logger(__name__)

TRAINING_HELP = textwrap.dedent(
    """
    FakeMark training entrypoint

    Usage:
      python main.py <config.yaml> [SpeechBrain/HyperPyYAML overrides]

    Maintained configs:
      config/fakemark_a.yaml  FakeMarkA / SEANet-style training
      config/fakemark_t.yaml  FakeMarkT / timbre-style training

    Common overrides:
      --base_path PATH
      --exp_name NAME
      --protocol_folder PATH
      --source_tracing_num_labels INT
      --batch_size INT
      --lr FLOAT
      --epochs INT

    Examples:
      python main.py config/fakemark_a.yaml
      python main.py config/fakemark_t.yaml --exp_name release_run

    Notes:
      - All remaining CLI flags are provided by SpeechBrain's argument parser.
      - Project-specific hyperparameters live in the YAML configs.
    """
).strip()


def infer_watermark_backend(modules: Any) -> str:
    """Infer the watermarking backend from the configured module graph.

    Args:
        modules: SpeechBrain module container for the active experiment.

    Returns:
        Watermark backend name, either ``"seanet"`` or ``"timbre"``.
    """
    has_decoder_path = hasattr(modules, "decoder") and hasattr(modules, "colprocessor")
    return "seanet" if has_decoder_path else "timbre"


class WMBrain(sb.core.Brain):
    """SpeechBrain training loop for watermark generation and attribution."""

    def init_optimizers(self) -> None:
        """Initialize generator and discriminator optimizers."""
        if self.opt_class is None:
            return

        opt_g_class, opt_d_class = self.opt_class
        generator_parameters = self._get_generator_parameters()
        self.optimizer_g = opt_g_class(generator_parameters)
        self.optimizer_d = opt_d_class(self.modules.discriminator.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer_g", self.optimizer_g)
            self.checkpointer.add_recoverable("optimizer_d", self.optimizer_d)
            self.checkpointer.add_recoverable(
                "global_step_counter",
                self.global_step_counter,
                custom_load_hook=load_step_counter,
                custom_save_hook=save_step_counter,
            )

        self.optimizers_dict = {
            "optimizer_g": self.optimizer_g,
            "optimizer_d": self.optimizer_d,
        }

    def _get_generator_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Collect trainable parameters for the generator-side optimization."""
        parameter_groups = [self.modules.encoder.parameters()]

        if hasattr(self.modules, "decoder"):
            parameter_groups.append(self.modules.decoder.parameters())
        if hasattr(self.modules, "colprocessor"):
            parameter_groups.append(self.modules.colprocessor.parameters())

        parameter_groups.append(self.modules.collaborator.parameters())
        return chain.from_iterable(parameter_groups)

    def _build_watermarked_audio(
        self, waveform: torch.Tensor, generator_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate watermark and watermarked waveform for the active backend."""
        backend = getattr(
            self.hparams,
            "watermark_backend",
            infer_watermark_backend(self.modules),
        )

        if backend == "seanet":
            watermark, watermarked_waveform = get_watermark_eval(
                waveform,
                generator_targets,
                self.modules.encoder,
                self.modules.decoder,
                self.modules.colprocessor,
            )
            return watermark, watermarked_waveform

        if backend == "timbre":
            watermarked_waveform, watermark = self.modules.encoder(
                waveform, generator_targets
            )
            return watermark, watermarked_waveform

        raise ValueError(
            f"Unsupported watermark backend '{backend}'. "
            "Expected one of: 'seanet', 'timbre'."
        )

    def compute_forward(
        self, batch: Dict[str, torch.Tensor], stage: sb.Stage
    ) -> Tuple[torch.Tensor, ...]:
        """Run the forward pass for a single batch."""
        waveform = batch["wav"].data.to(device=self.device, non_blocking=True)
        waveform = waveform.unsqueeze(1)
        collaborator_targets = batch["logit"].to(device=self.device, non_blocking=True)

        if stage == sb.Stage.TRAIN:
            generator_targets = torch.randint(
                0,
                self.hparams.source_tracing_num_labels,
                (waveform.shape[0],),
                device=self.device,
            )
        else:
            generator_targets = collaborator_targets

        watermark, watermarked_waveform = self._build_watermarked_audio(
            waveform, generator_targets
        )

        scores_fake, feats_fake = self.modules.discriminator(
            watermarked_waveform.detach()
        )
        scores_real, feats_real = self.modules.discriminator(waveform)

        return (
            waveform,
            watermarked_waveform,
            watermark,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            collaborator_targets,
            generator_targets,
        )

    def compute_objectives(
        self,
        predictions: Tuple[torch.Tensor, ...],
        batch: Dict[str, torch.Tensor],
        stage: sb.Stage,
    ) -> Dict[str, torch.Tensor]:
        """Compute all optimization objectives for the current batch."""
        (
            waveform,
            watermarked_waveform,
            watermark,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            collaborator_targets,
            generator_targets,
        ) = predictions

        generator_recon_loss = self.hparams.generator_loss(
            stage,
            watermarked_waveform,
            waveform,
            scores_fake,
            feats_fake,
            feats_real,
        )
        adversarial_loss = self.hparams.discriminator_loss(scores_fake, scores_real)
        loudness_loss = self.hparams.loudness_loss(out_sig=watermark, ref_sig=waveform)
        spectral_shape_loss = freq_mag_loss(watermark, waveform)
        watermark_l1_loss = torch.nn.functional.l1_loss(
            watermark, torch.zeros_like(watermark)
        )

        collaborator_predictions, _ = self.modules.collaborator(
            batchwise_random_perturbation(watermarked_waveform)
        )
        generator_collaborator_loss = torch.nn.functional.cross_entropy(
            collaborator_predictions, generator_targets
        )

        collaborator_loss = self._collaborator_objective(
            waveform, collaborator_targets
        )

        objectives = {
            "loud_loss": loudness_loss,
            "l1_loss": watermark_l1_loss,
            "g_collab_loss": generator_collaborator_loss,
            "collab_loss": collaborator_loss,
            "ssn_loss": spectral_shape_loss,
            **generator_recon_loss,
            **adversarial_loss,
            "valid_log_loss": generator_collaborator_loss + collaborator_loss,
        }
        self.last_loss_stats[stage] = scalarize(objectives)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                ids=batch["id"],
                predictions=collaborator_predictions.argmax(dim=1),
                targets=generator_targets,
            )

        return objectives

    def _collaborator_objective(
        self, waveform: torch.Tensor, collaborator_targets: torch.Tensor
    ) -> torch.Tensor:
        """Train the collaborator on perturbed clean audio."""
        collaborator_predictions, _ = self.modules.collaborator(
            batchwise_random_perturbation(waveform.detach())
        )
        return torch.nn.functional.cross_entropy(
            collaborator_predictions, collaborator_targets
        )

    def on_fit_start(self) -> None:
        """Initialize training state before the first optimization step."""
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        self.global_step_counter = StepCounter(step=0)
        self.last_ckpt_time = time.time()
        super().on_fit_start()

    def fit_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Update discriminator first, then generator-side modules."""
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        (
            waveform,
            watermarked_waveform,
            watermark,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            collaborator_targets,
            generator_targets,
        ) = outputs

        self.last_batch = (waveform, watermarked_waveform)

        discriminator_objectives = self.compute_objectives(
            outputs, batch, sb.Stage.TRAIN
        )
        discriminator_loss = discriminator_objectives["D_loss"]
        self.optimizer_d.zero_grad()
        discriminator_loss.backward()
        self.optimizer_d.step()

        scores_fake, feats_fake = self.modules.discriminator(watermarked_waveform)
        scores_real, feats_real = self.modules.discriminator(waveform)
        generator_outputs = (
            waveform,
            watermarked_waveform,
            watermark,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            collaborator_targets,
            generator_targets,
        )
        objectives = self.compute_objectives(generator_outputs, batch, sb.Stage.TRAIN)

        generator_loss = self._compute_generator_loss(objectives)
        self.optimizer_g.zero_grad()
        generator_loss.backward()
        self.optimizer_g.step()

        objectives["generator_loss"] = generator_loss.detach()
        self.on_fit_batch_end(objectives)
        self.global_step_counter.step += 1
        return objectives["g_collab_loss"].detach()

    def _compute_generator_loss(
        self, objectives: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combine weighted loss terms for generator-side optimization."""
        if self.global_step_counter.step < self.hparams.msg_warmup:
            return (
                self.hparams.lambda_g_col * objectives["g_collab_loss"]
                + self.hparams.lambda_col * objectives["collab_loss"]
            )

        return (
            self.hparams.lambda_gen * objectives["G_loss"]
            + self.hparams.lambda_loud * objectives["loud_loss"]
            + self.hparams.lambda_l1 * objectives["l1_loss"]
            + self.hparams.lambda_g_col * objectives["g_collab_loss"]
            + self.hparams.lambda_col * objectives["collab_loss"]
            + self.hparams.lambda_ssn * objectives["ssn_loss"]
        )

    def on_fit_batch_end(self, objectives: Dict[str, torch.Tensor]) -> None:
        """Apply schedulers and emit per-step logs."""
        self.hparams.scheduler_generator(self.optimizer_g)
        self.hparams.scheduler_discriminator(self.optimizer_d)

        if not (
            hasattr(self.hparams, "log_interval")
            and self.global_step_counter.step % self.hparams.log_interval == 0
        ):
            return

        log_stats = {
            key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in objectives.items()
        }
        log_stats["global_step"] = self.global_step_counter.step
        log_stats["lr_g"] = self.optimizer_g.param_groups[0]["lr"]
        log_stats["lr_d"] = self.optimizer_d.param_groups[0]["lr"]
        log_stats["avg_loss"] = self.avg_train_loss
        log_stats["snr"] = self._compute_last_batch_snr()

        if hasattr(self, "valid_epoch_loss"):
            log_stats["valid_epoch_loss"] = self.valid_epoch_loss

        if hasattr(self, "time_last_log"):
            log_stats["run_time"] = time.time() - self.time_last_log
        self.time_last_log = time.time()

        if sb.utils.distributed.if_main_process():
            self.hparams.train_steps_logger.log_stats(stats_meta=log_stats)

        self.save_inference_sample()

    def _compute_last_batch_snr(self) -> torch.Tensor:
        """Estimate average SNR for the most recently processed batch."""
        if self.last_batch is None:
            return torch.tensor(0.0, device=self.device)

        waveform, watermarked_waveform = self.last_batch
        with torch.no_grad():
            noise = waveform - watermarked_waveform
            snr_per_sample = 10 * torch.log10(
                torch.sum(waveform**2, dim=-1)
                / (torch.sum(noise**2, dim=-1) + 1e-8)
            )
        return snr_per_sample.mean()

    def fit(
        self,
        epoch_counter: Any,
        train_set: Any,
        valid_set: Any = None,
        progressbar: Optional[bool] = None,
        train_loader_kwargs: Optional[Dict[str, Any]] = None,
        valid_loader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run the customized SpeechBrain training loop."""
        train_loader_kwargs = train_loader_kwargs or {}
        valid_loader_kwargs = valid_loader_kwargs or {}

        if not isinstance(train_set, (DataLoader, LoopedLoader)):
            train_set = self.make_dataloader(
                train_set,
                stage=sb.Stage.TRAIN,
                **train_loader_kwargs,
            )
        if valid_set is not None and not isinstance(valid_set, (DataLoader, LoopedLoader)):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        enable_progressbar = progressbar and sb.utils.distributed.if_main_process()

        for epoch in epoch_counter:
            self._fit_train_customized(
                train_set=train_set,
                valid_set=valid_set,
                epoch=epoch,
                enable=enable_progressbar,
                valid_step=self.hparams.valid_step,
            )

        self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable_progressbar)

    def _fit_train_customized(
        self,
        train_set: Any,
        valid_set: Any,
        epoch: int,
        enable: bool,
        valid_step: int = 100,
    ) -> None:
        """Train for one epoch with periodic intra-epoch validation."""
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as progress:
            if self.profiler is not None:
                self.profiler.start()

            for batch in progress:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break

                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                progress.set_postfix(train_loss=self.avg_train_loss)

                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info("The profiler finished, training is stopped.")
                        self.profiler.stop()
                        raise SystemExit(0)

                elapsed_hours = (time.time() - self.last_ckpt_time) / 3600
                if elapsed_hours >= self.hparams.ckpt_interval_hours:
                    self._save_intra_epoch_ckpt()
                    self.last_ckpt_time = time.time()

                if (
                    self.global_step_counter.step % valid_step == 0
                    and self.global_step_counter.step > self.hparams.msg_warmup
                ):
                    self._fit_valid_customized(valid_set, epoch)

        self.zero_grad(set_to_none=True)
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0

    def _fit_valid_customized(self, valid_set: Any, epoch: int) -> None:
        """Run validation during training."""
        if valid_set is None:
            return

        self.on_stage_start(sb.Stage.VALID, epoch)
        self.modules.eval()
        avg_valid_loss = 0.0
        self.step = 0

        with torch.no_grad():
            for batch in valid_set:
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                avg_valid_loss = self.update_average(loss, avg_valid_loss)

        self.step = 0
        self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)
        self.modules.train()

    def evaluate_batch(
        self, batch: Dict[str, Any], stage: sb.Stage
    ) -> torch.Tensor:
        """Evaluate one batch and return the validation logging loss."""
        predictions = self.compute_forward(batch, stage=stage)
        objectives = self.compute_objectives(predictions, batch, stage=stage)

        if stage == sb.Stage.VALID:
            batch_valid_loss = objectives["valid_log_loss"].detach().cpu().item()
            self.valid_epoch_loss = (
                self.valid_epoch_loss + batch_valid_loss
                if hasattr(self, "valid_epoch_loss")
                else batch_valid_loss
            )

        return objectives["valid_log_loss"].detach().cpu()

    def on_stage_start(self, stage: sb.Stage, epoch: int) -> None:
        """Reset stage-specific metrics before validation/test."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = sb.utils.metric_stats.ClassificationStats()
            if stage == sb.Stage.VALID:
                self.valid_epoch_loss = 0.0

    def on_stage_end(self, stage: sb.Stage, stage_loss: Any, epoch: int) -> None:
        """Log stage metrics and update checkpoints."""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            return

        if stage != sb.Stage.VALID:
            return

        stage_accuracy = self.error_metrics.summarize(field="accuracy")
        stage_stats = {
            "loss": stage_loss,
            "acc": stage_accuracy * 100,
        }
        self.hparams.train_stage_logger.log_stats(
            stats_meta={
                "Epoch": epoch,
                "Step": self.global_step_counter.step,
                "lr_g": self.optimizer_g.param_groups[-1]["lr"],
                "lr_d": self.optimizer_d.param_groups[-1]["lr"],
            },
            train_stats=None,
            valid_stats=stage_stats,
        )
        self.checkpointer.save_and_keep_only(
            end_of_epoch=True,
            num_to_keep=1,
            meta={
                "Loss": stage_stats["loss"],
                "BitwiseAccuracy": stage_stats["acc"],
            },
            min_keys=["Loss"],
        )

    def save_inference_sample(self) -> None:
        """Persist the latest clean and watermarked samples for inspection."""
        if self.last_batch is None:
            return

        target_dir = Path(self.hparams.wav_folder) / str(self.global_step_counter.step)
        target_dir.mkdir(parents=True, exist_ok=True)

        waveform, watermarked_waveform = self.last_batch
        with torch.no_grad():
            for index in range(waveform.size(0)):
                sf.write(
                    str(target_dir / f"carrier_{index}.wav"),
                    waveform[index].detach().cpu().squeeze(0).numpy(),
                    self.hparams.sample_rate,
                )
                sf.write(
                    str(target_dir / f"watermarked_{index}.wav"),
                    watermarked_waveform[index].detach().cpu().squeeze(0).numpy(),
                    self.hparams.sample_rate,
                )


def maybe_print_training_help(argv: list[str]) -> None:
    """Print user-facing help before delegating to SpeechBrain parsing."""
    if not argv or argv[0] in {"-h", "--help"}:
        print(TRAINING_HELP)
        raise SystemExit(0)


def load_hparams_and_runtime(
    argv: list[str],
) -> tuple[str, dict[str, Any], str, Any]:
    """Parse CLI arguments and load HyperPyYAML configuration.

    Args:
        argv: Raw command-line arguments excluding the program name.

    Returns:
        Tuple containing the YAML path, SpeechBrain runtime options, override
        string, and loaded hyperparameter object.
    """
    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    return hparams_file, run_opts, overrides, hparams


def prepare_datasets(
    hparams: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Build datasets and data-loaders using the merged-STC data pipeline.

    Args:
        hparams: Experiment hyperparameter mapping.

    Returns:
        Tuple of ``(train_data, valid_data, train_loader_kwargs,
        valid_loader_kwargs)``.
    """
    return _dataio_prepare(hparams)


def main(argv: Optional[list[str]] = None) -> None:
    """Launch training for the selected experiment configuration.

    Args:
        argv: Optional command-line argument list. When omitted, the values from
            ``sys.argv`` are used.
    """
    argv = sys.argv[1:] if argv is None else argv
    maybe_print_training_help(argv)
    hparams_file, run_opts, overrides, hparams = load_hparams_and_runtime(argv)
    set_random_seed(hparams["seed"])

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_data, valid_data, train_loader_kwargs, valid_loader_kwargs = prepare_datasets(
        hparams
    )

    brain = WMBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    logger.info(
        "Starting training with watermark backend: %s",
        hparams.get(
            "watermark_backend",
            infer_watermark_backend(hparams["modules"]),
        ),
    )

    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=valid_loader_kwargs,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
