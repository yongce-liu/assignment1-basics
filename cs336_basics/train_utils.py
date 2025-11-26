import torch
import numpy as np
import os
import typing
import pickle as pkl
import argparse
import logging
import time
from cs336_basics.models import (
    Transformer,
    AdamW,
    lr_cosine_schedule,
    gradient_clip,
    cross_entropy_loss,
)


def dataloader(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"):
    # Numpy implements this through
    # np.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), which
    # will return a numpy array-like object that loads the entries on-demand as you access them.
    x = torch.from_numpy(x)

    # Maximum valid starting index (need context_length + 1 tokens for input and target)
    max_start_idx = len(x) - context_length

    # Randomly sample starting indices for each sequence in the batch
    start_indices = torch.randint(0, max_start_idx, (batch_size,))

    # Create batch by gathering sequences starting at each sampled index
    inputs = torch.stack([x[i : i + context_length] for i in start_indices])
    targets = torch.stack([x[i + 1 : i + context_length + 1] for i in start_indices])

    return inputs.to(device), targets.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    ckp = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration)
    f = open(out, "wb") if isinstance(out, (str, os.PathLike)) else out
    pkl.dump(ckp, f)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    f = open(src, "rb") if isinstance(src, (str, os.PathLike)) else src
    ckp = pkl.load(f)
    model.load_state_dict(ckp["model"])
    optimizer.load_state_dict(ckp["optimizer"])
    return ckp["iteration"]


class TrainingArgs(argparse.Namespace):
    train_path: str
    valid_path: str
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    rope_theta: float
    batch_size: int
    lr: float
    max_iters: int
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    grad_clip: float
    output_dir: str
    log_interval: int
    eval_interval: int
    eval_iters: int
    wandb_project: str | None
    resume_from: str | None


def get_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description="Train a Transformer model")

    # Data arguments
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data (numpy memmap)")
    parser.add_argument("--valid_path", type=str, required=True, help="Path to validation data (numpy memmap)")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000, help="Cosine cycle iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # Checkpointing and Logging
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations for evaluation")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args(namespace=TrainingArgs())


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = dataloader(data, batch_size, context_length, device)
        logits = model(X)
        # Flatten for cross_entropy_loss
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def train():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize WandB
    if args.wandb_project:
        try:
            import wandb

            wandb.init(project=args.wandb_project, config=args)
        except ImportError:
            logger.warning("wandb not installed, skipping logging to wandb")
            args.wandb_project = None

    # Load data
    # Assuming data is stored as uint16 (common for vocab size < 65536)
    # If vocab size > 65536, might need int32 or int64.
    # We'll try to infer or assume uint16 for now as it's standard for GPT-2/Llama tokenizers.
    # But np.memmap needs a dtype. Let's assume np.uint16.
    try:
        train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
        valid_data = np.memmap(args.valid_path, dtype=np.uint16, mode="r")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Valid data size: {len(valid_data)}")

    # Initialize model
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        context_length=args.context_length,
        device=device,
        dtype=torch.float32,  # Or bfloat16 if supported
    )
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    # Training loop
    t0 = time.time()
    for iter_num in range(start_iter, args.max_iters):
        # Determine learning rate
        lr = lr_cosine_schedule(iter_num, args.lr, args.lr * 0.1, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        X, Y = dataloader(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(X)
        loss = cross_entropy_loss(logits.view(-1, args.vocab_size), Y.view(-1))

        # Backward pass
        model.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0.0:
            gradient_clip(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            logger.info(f"Iter {iter_num}: loss {loss.item():.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")
            if args.wandb_project:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "iter": iter_num})

        # Evaluation
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, valid_data, args.batch_size, args.context_length, args.eval_iters, device)
            logger.info(f"Iter {iter_num}: val loss {val_loss:.4f}")
            if args.wandb_project:
                wandb.log({"val/loss": val_loss, "iter": iter_num})

            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"ckpt_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.output_dir, "ckpt_final.pt"))
    logger.info("Training complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    train()
