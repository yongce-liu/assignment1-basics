from pathlib import Path
from pprint import pprint

from .train_utils import TrainingArgs

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

if __name__ == "__main__":
    params = dict(
        name="tiny-stories",
        train_path=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-train-tokens.npy",
        valid_path=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-valid-tokens.npy",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_heads=4,
        d_ff=1344,
        num_layers=4,
        rope_theta=10000.0,
        seed=0,
        batch_size=32,
        learning_rate=5e-4,
        adamw_wd=0.1,
        adamw_betas=(0.9, 0.95),
        max_iters=5000,
        warmup_iters=500,
        cosine_cycle_iters=5000,
        grad_clip=1.0,
        output_dir=PROJECT_ROOT + "/outputs/tiny-stories",
        log_interval=10,
        eval_interval=500,
        eval_iters=200,
        wandb_entity="yongce_llm",
        resume=None,
    )
    args = TrainingArgs(**params)
    print("Training with the following parameters:")
    pprint(params, indent=2, width=80)
