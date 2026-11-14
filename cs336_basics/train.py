import torch
import numpy as np
import os
import argparse
from cs336_basics.nn.transformer_lm import TranformerLM
from cs336_basics.optim import AdamW
import cs336_basics.utils as utils
from cs336_basics.lr_scheduler import CosineScheduler, ConstantScheduler
from cs336_basics.nn.loss import cross_entropy_loss
from cs336_basics.data.utils import data_loading
import wandb
import time
import math
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    run: wandb.Run,
):
    model.train()
    start_time = time.perf_counter()
    interval_start = start_time

    # train_dset = np.load(args.training_dset, mmap_mode="r")
    train_dset = np.random.random_integers(low=1, high=1024, size=(1_000_000,))
    if args.validation_dset:
        valid_set = np.load(args.validation_dset, mmap_mode="r")

    if args.load_ckpt:
        iter = utils.load_checkpoint(args.load_ckpt)
    else:
        iter = 1

    logger.info(
        "Starting training | device=%s | dtype=%s | parameters=%s | tokens=%s",
        device,
        next(model.parameters()).dtype,
        f"{sum(p.numel() for p in model.parameters()):,}",
        f"{train_dset.size:,}",
    )
    loss_sum = 0
    num_batch = 0
    num_tokens = 0

    while iter < args.max_iters:

        optimizer.zero_grad()
        x, y = data_loading(
            train_dset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        y_hat = model(x)

        loss = cross_entropy_loss(y_hat, y)

        loss_sum += loss.detach().item()
        num_batch += 1
        num_tokens += y.numel()

        loss.backward()
        gnorm = utils.gradient_clipping(model.parameters(), args.clip_norm)

        optimizer.step()

        metrics = {
            "train/step": iter,
            "train/gnorm": gnorm.item(),
        }

        if iter % args.log_every == 0 or iter + 1 >= args.max_iters:
            now = time.perf_counter()
            elapsed_minutes = (now - start_time) / 60
            tokens_per_second = num_tokens / (now - interval_start)

            avg_loss = loss_sum / num_batch
            perplexity = math.exp(avg_loss)

            metrics.update(
                {
                    "train/elapsed_minutes": elapsed_minutes,
                    "train/loss": avg_loss,
                    "train/loss_vs_time": avg_loss,
                    "train/perplexity": perplexity,
                    "train/tokens_per_second": tokens_per_second,
                }
            )

            logger.info(
                "step=%d | loss=%.4f | ppl=%.2f | gnorm=%.3f | "
                "elapsed=%.2f min | tokens/sec=%.0f",
                iter,
                avg_loss,
                perplexity,
                gnorm.item(),
                elapsed_minutes,
                tokens_per_second,
            )

            loss_sum = 0.0
            num_batch = 0
            num_tokens = 0
            interval_start = now

        run.log(metrics)
        if args.save_every > 0 and iter % args.save_every == 0:
            path = os.path.join(args.output_dir, f"ckpt_{iter}.pt")
            utils.save_checkpoint(model, optimizer, iter, out=path)
            logger.info("Saved checkpoint: %s", path)

        iter += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--vocab_size", default=50_257, type=int)
    argparser.add_argument("--context_length", default=1024, type=int)
    argparser.add_argument("--num_layers", default=48, type=int)
    argparser.add_argument("--d_model", default=1600, type=int)
    argparser.add_argument("--nheads", default=25, type=int)
    argparser.add_argument("--d_ff", default=4_288, type=int)
    argparser.add_argument("--rope_theta", default=1e4)
    argparser.add_argument("--lr_schedule", default="None", choices=["None", "cosine"])
    argparser.add_argument("--lr_schedule_cosince_cycle", default=10_000, type=int)
    argparser.add_argument("--lr_min", default=1e-5, type=float)
    argparser.add_argument("--lr_warmup", default=1_000, type=int)
    argparser.add_argument("--lr_max", default=1e-4, type=float)
    argparser.add_argument("--lr", default=1e-4, type=float)

    argparser.add_argument("--beta1", default=0.9, type=float)
    argparser.add_argument("--beta2", default=0.999, type=float)
    argparser.add_argument("--weight_decay", default=0.1, type=float)

    argparser.add_argument("--device", default="cpu", type=str)
    argparser.add_argument("--dtype", default="bfloat16", type=str)

    argparser.add_argument("--output_dir")
    argparser.add_argument("--load_ckpt", default=None)
    argparser.add_argument("--training_dset")
    argparser.add_argument("--validation_dset", default=None)
    argparser.add_argument("--save_every", default=-1, type=int)
    argparser.add_argument("--validate_every", default=-1, type=int)

    argparser.add_argument("--wandb_project", default="cs336")
    argparser.add_argument("--wandb_name", default=None)
    argparser.add_argument("--log_every", type=int, default=10)

    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--max_iters", default=1e6, type=int)
    argparser.add_argument("--clip_norm", default=1.0, type=float)

    argparser.add_argument("--config")
    args = utils.load_yaml(argparser)
    device = torch.device(device=args.device)

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float16

    model = TranformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.nheads,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )

    if args.lr_schedule == "cosine":
        scheduler = CosineScheduler(
            args.lr_max, args.lr_min, args.lr_warmup, args.lr_schedule_cosince_cycle
        )
    else:
        scheduler = ConstantScheduler(args.lr)

    optimizer = AdamW(
        lr=args.lr,
        params=model.parameters(),
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        scheulder=scheduler,
    )

    entity, project = args.wandb_project.split("/", 1)
    wandb_name = Path(args.config).stem if args.wandb_name is None else args.wandb_name
    with wandb.init(
        entity=entity,
        project=project,
        name=wandb_name,
        config=vars(args),  # Records your resolved YAML/CLI configuration.
    ) as run:
        os.makedirs(args.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(args.output_dir, "train.log"),
                    encoding="utf-8",
                ),
            ],
        )
        run.define_metric("train/loss", step_metric="train/step")
        run.define_metric("train/loss_vs_time", step_metric="train/elapsed_minutes")
        run.define_metric("train/perplexity", step_metric="train/step")
        run.define_metric("train/gnorm", step_metric="train/step")

        train(args, model, optimizer, device=device, run=run)
