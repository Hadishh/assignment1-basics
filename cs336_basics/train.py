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


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device=torch.device,
):
    train_dset = np.load(args.training_dset, mmap_mode="r")
    if args.validation_dset:
        valid_set = np.load(args.training_dset, mmap_mode="r")

    if args.load_ckpt:
        iter = utils.load_checkpoint(args.load_ckpt)
    else:
        iter = 1

    while iter < args.max_iters:
        x, y = data_loading(
            train_dset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        y_hat = model(x)

        loss = cross_entropy_loss(y_hat, y)
        loss.backward()
        utils.gradient_clipping(model.parameters(), args.clip_norm)

        optimizer.step()

        if iter % args.save_every == 0:
            path = os.path.join(args.output_dir, f"ckpt_{iter}.pt")
            utils.save_checkpoint(model, optimizer, iter, out=path)

        iter += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--vocab_size", default=50_257, type=int)
    argparser.add_argument("--context_legth", default=1024, type=int)
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

    argparser.add_argument("--output_dir", required=True)
    argparser.add_argument("--load_ckpt", default=None)
    argparser.add_argument("--training_dset", required=True)
    argparser.add_argument("--validation_dset", default=None)
    argparser.add_argument("--save_every", default=-1, type=int)
    argparser.add_argument("--validate_every", default=-1, type=int)

    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--max_iters", default=1e6, type=int)
    argparser.add_argument("--clip_norm", default=1.0, type=float)

    args = argparser.parse_args()
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
        params=model.parameters(),
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        scheulder=scheduler,
    )
