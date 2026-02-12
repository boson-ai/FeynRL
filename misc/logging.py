import os
import logging

def setup_logging(rank: int, log_level: str = "INFO", exp_name: str = "") -> logging.Logger:
    '''
        Setup logging configuration. Only rank 0 logs to console to avoid duplicate messages.
    '''
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format with timestamp and rank
    formatter = logging.Formatter(
        fmt=f"[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_viz(config, tracking_uri: str, rank: int):
    """
    Setup experiment tracking.
    Supports 'mlflow' and 'wandb'.
    Only rank 0 logs metrics.
    """
    if rank != 0:
        return None

    logger_type = getattr(config.run, "logger_type", "mlflow").lower()

    if logger_type == "mlflow":
        import mlflow
        # MLflow setup (existing code)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.run.project_name)
        run = mlflow.start_run(run_name=config.run.experiment_id)

        mlflow.log_params({
            "alg_name": config.train.alg_name,
            "model_name": config.model.name,
            "learning_rate": config.train.lr,
            "train_batch_size_per_gpu": config.train.train_batch_size_per_gpu,
            "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
            "total_epochs": config.train.total_number_of_epochs,
            "seed": config.run.seed,
        })

        if config.run.method == "sl":
            mlflow.log_params({
                "micro_batches_per_epoch": config.train.micro_batches_per_epoch,
            })
        elif config.run.method == "rl":
            mlflow.log_params({
                "train_steps_per_epoch": config.train.train_steps_per_epoch,
                "n_samples": config.rollout.n_samples,
                "max_tokens": config.rollout.max_tokens,
                "kl_coeff": config.train.kl_coeff,
                "clip_low": config.train.clip_low,
                "clip_high": config.train.clip_high,
                "entropy_coeff": config.train.entropy_coeff,
                "training_gpus": config.run.training_gpus,
                "rollout_gpus": config.run.rollout_gpus,
            })
        return run

    elif logger_type == "wandb":
        import wandb

        # Load API key from file
        key_path = "./.wandb_key"
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"W&B key file not found: {key_path}")
        with open(key_path, "r") as f:
            api_key = f.read().strip()

        # Set environment variable so wandb can authenticate
        os.environ["WANDB_API_KEY"] = api_key

        run = wandb.init(
            project=config.run.project_name,
            name=config.run.experiment_id,
            config={**config.train.__dict__, **config.run.__dict__},
        )
        return run

    else:
        raise ValueError(f"Unknown logger type: {logger_type}")