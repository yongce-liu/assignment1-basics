from pprint import pprint

import colorlog

from cs336_basics.train_utils import PROJECT_ROOT, TrainingArgs, train

if __name__ == "__main__":
    params = dict(
        name="tiny-stories",
        # train_path=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-train-tokens.npy",
        train_path=PROJECT_ROOT + "/data/TinyStoriesV2-GPT4-valid-tokens.npy",
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

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logger = colorlog.getLogger(args.name)
    logger.addHandler(handler)
    logger.setLevel(20)

    logger.info("Training with the following parameters:")
    pprint(params, indent=2, width=80, sort_dicts=False)

    logger.info("Starting training...")
    train(args=args, logger=logger)
    logger.info("Training completed.")
