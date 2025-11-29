import argparse
import logging
import os
import pickle as pkl
import time
import typing
from pathlib import Path

import numpy as np
import torch

from cs336_basics.models import (
    AdamW,
    Transformer,
    cross_entropy_loss,
    gradient_clip,
    lr_cosine_schedule,
)

PROJECT_ROOT: str = str(Path(__file__).parent.parent)


def set_seed(seed):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader(
    x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu", dtype: torch.dtype = torch.long
) -> tuple[torch.Tensor, torch.Tensor]:
    # Numpy implements this through
    # np.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), which
    # will return a numpy array-like object that loads the entries on-demand as you access them.
    x = torch.from_numpy(x)

    # Maximum valid starting index (need context_length + 1 tokens for input and target)
    max_start_idx = len(x) - context_length

    # Randomly sample starting indices for each sequence in the batch
    start_indices = torch.randint(0, max_start_idx, (batch_size,))

    # Create batch by gathering sequences starting at each sampled index
    inputs = torch.stack([x[i : i + context_length] for i in start_indices]).to(dtype=dtype, device=device)
    targets = torch.stack([x[i + 1 : i + context_length + 1] for i in start_indices]).to(dtype=dtype, device=device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration)
    f = open(out, "wb") if isinstance(out, (str, os.PathLike)) else out
    pkl.dump(ckpt, f)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    f = open(src, "rb") if isinstance(src, (str, os.PathLike)) else src
    ckpt = pkl.load(f)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


def inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    min_p: float = 0.0,
    eps: float = 1e-8,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, seq_len)
    assert len(input_ids.shape) == 2, "input_ids should be of shape (B, seq_len)"
    is_finished = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
    output_ids = torch.zeros((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
    model.eval()
    for idx in range(max_new_tokens):
        # Get the logits of the next token
        logits = model(input_ids)  # (B, seq_len, vocab_size)
        #################################################################
        # temperature scaling
        logits = logits[:, -1, :] / (temperature + eps)  # (B, vocab_size)
        # softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
        # nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = sorted_indices[cumulative_probs < min_p] if sorted_probs[0] > min_p else sorted_indices[1:]
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize the probabilities
        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1) # assume the index is equal to token id
        ###############################################################
        if end_token_id is not None:
            is_finished = is_finished | (next_token.squeeze(-1) == end_token_id)
            # Append the next token only for unfinished sequences
            next_token = next_token.masked_fill(is_finished.unsqueeze(-1), end_token_id)
        # Append the next token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)  # (B, seq_len++)
        output_ids[:, idx] = next_token.squeeze(-1)
        if is_finished.all():
            break

    model.train()
    return output_ids


class TrainingArgs(argparse.Namespace):
    name: str
    train_path: str
    valid_path: str
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    rope_theta: float
    seed: int
    batch_size: int
    learning_rate: float
    adamw_betas: tuple[float, float]
    max_iters: int
    warmup_iters: int
    cosine_cycle_iters: int
    adamw_wd: float
    grad_clip: float
    output_dir: str
    log_interval: int
    eval_interval: int
    eval_iters: int
    wandb_entity: str | None
    resume: str | None


def get_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description="Train a Transformer model")

    parser.add_argument("--name", type=str, default="tiny-stories", help="project name")
    # Data arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-train-tokens.npy",
        help="Path to training data (numpy memmap)",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-valid-tokens.npy",
        help="Path to validation data (numpy memmap)",
    )
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")

    # Training arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Max learning rate")
    parser.add_argument("--adamw_wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--adamw_betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000, help="Cosine cycle iterations")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # Checkpointing and Logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default=PROJECT_ROOT + "/outputs/tiny-stories",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations for evaluation")
    parser.add_argument("--wandb_entity", type=str, default="yongce_llm", help="WandB entity name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args(namespace=TrainingArgs())


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = dataloader(data, batch_size, context_length, device)
        logits = model(X)
        # Flatten for cross_entropy_loss
        # assume idx = token id for Y
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def train(args: TrainingArgs | None = None, logger: logging.Logger | None = None):
    args = get_args() if args is None else args
    if logger is None:
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(args.name)

    logger.info(f"Setting random seed as {args.seed}")
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize WandB
    if args.wandb_entity:
        try:
            import wandb

            wandb.init(entity=args.wandb_entity, project=args.name, config=args)
        except ImportError:
            logger.warning("wandb not installed, skipping logging to wandb")
            args.wandb_entity = None

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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=args.adamw_betas, weight_decay=args.adamw_wd)

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        logger.info(f"Resuming from {args.resume}, starting at iteration {start_iter}")

    # Training loop
    t0 = time.time()
    for iter_num in range(start_iter, args.max_iters):
        # Determine learning rate
        lr = lr_cosine_schedule(
            iter_num, args.learning_rate, args.learning_rate * 0.1, args.warmup_iters, args.cosine_cycle_iters
        )
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
            if args.wandb_entity:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "iter": iter_num})

        # Evaluation
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, valid_data, args.batch_size, args.context_length, args.eval_iters, device)
            logger.info(f"Iter {iter_num}: val loss {val_loss:.4f}")
            if args.wandb_entity:
                wandb.log({"val/loss": val_loss, "iter": iter_num})

            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"ckpt_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.output_dir, "ckpt_final.pt"))
    logger.info("Training complete")


if __name__ == "__main__":
    train(logger)
