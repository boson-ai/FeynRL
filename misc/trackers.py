import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters of the experiment."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training or evaluation metrics."""
        pass

    @abstractmethod
    def finish(self):
        """Signal that the experiment run has ended."""
        pass

class MLFlowTracker(ExperimentTracker):
    def __init__(self, config, tracking_uri: str):
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(config.run.project_name)
        self.run = self.mlflow.start_run(run_name=config.run.experiment_id)
        
        # Log default params
        params = {
            "alg_name": config.train.alg_name,
            "model_name": config.model.name,
            "learning_rate": config.train.lr,
            "train_batch_size_per_gpu": config.train.train_batch_size_per_gpu,
            "total_epochs": config.train.total_number_of_epochs,
            "seed": config.run.seed,
        }
        
        # Add grad accumulation if available
        grad_acc = getattr(config.train, "gradient_accumulation_steps", None)
        if grad_acc is not None:
            params["gradient_accumulation_steps"] = grad_acc

        if config.run.method in ["sl", "cl"]:
            params["micro_batches_per_epoch"] = getattr(config.train, "micro_batches_per_epoch", None)
            if config.run.method == "cl":
                params["beta"] = getattr(config.train, "cl_beta", None)

        elif config.run.method == "rl":
            params.update({
                "train_steps_per_epoch": getattr(config.train, "train_steps_per_epoch", None),
                "n_samples": getattr(config.rollout, "n_samples", None),
                "max_tokens": getattr(config.rollout, "max_tokens", None),
                "kl_coeff": getattr(config.train, "kl_coeff", None),
                "clip_low": getattr(config.train, "clip_low", None),
                "clip_high": getattr(config.train, "clip_high", None),
                "entropy_coeff": getattr(config.train, "entropy_coeff", None),
                "training_gpus": getattr(config.run, "training_gpus", None),
                "rollout_gpus": getattr(config.run, "rollout_gpus", None),
            })
        
        # Filter None values
        params = {k: v for k, v in params.items() if v is not None}
        self.log_params(params)

    def log_params(self, params: Dict[str, Any]):
        self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.mlflow.log_metrics(metrics, step=step)

    def finish(self):
        self.mlflow.end_run()

class WandBTracker(ExperimentTracker):
    def __init__(self, config):
        import wandb
        self.wandb = wandb
        
        # Load API key from file
        key_path = "./.wandb_key"
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = api_key
        
        # Initialize wandb with combined config
        # We try to convert config objects to dicts where possible
        wandb_config = {}
        for section in ["train", "run", "model", "rollout", "data", "reward"]:
            if hasattr(config, section):
                sec_obj = getattr(config, section)
                if hasattr(sec_obj, "__dict__"):
                    wandb_config.update(sec_obj.__dict__)
                elif hasattr(sec_obj, "dict"): # pydantic v1
                    wandb_config.update(sec_obj.dict())
                elif hasattr(sec_obj, "model_dump"): # pydantic v2
                    wandb_config.update(sec_obj.model_dump())

        self.run = self.wandb.init(
            project=config.run.project_name,
            name=config.run.experiment_id,
            config=wandb_config,
        )

    def log_params(self, params: Dict[str, Any]):
        self.wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # Ensure all metrics are wandb-compatible (floats/ints)
        formatted_metrics = {}
        for k, v in metrics.items():
            try:
                formatted_metrics[k] = float(v)
            except (ValueError, TypeError):
                formatted_metrics[k] = v
        
        self.wandb.log(formatted_metrics, step=step)

    def finish(self):
        self.wandb.finish()

class TrackerRegistry:
    """
    Registry for experiment trackers.

    Notes:
    - Only rank 0 will have a tracker.
    - Default tracker is mlflow.
    """
    _trackers: Dict[str, Type[ExperimentTracker]] = {
        "mlflow": MLFlowTracker,
        "wandb": WandBTracker
    }

    @classmethod
    def register(cls, name: str, tracker_class: Type[ExperimentTracker]):
        cls._trackers[name.lower()] = tracker_class

    @classmethod
    def get_tracker(cls, config, rank: int) -> Optional[ExperimentTracker]:
        if rank != 0:
            return None
        
        # extract logger type from config, default to mlflow
        logger_type = getattr(config.run, "logger_type", "mlflow").lower()
        
        if logger_type not in cls._trackers:
            print(f"Warning: Unknown logger type '{logger_type}'. No external tracking will be used.")
            return None
            
        tracker_class = cls._trackers[logger_type]
        
        try:
            if logger_type == "mlflow":
                return tracker_class(config, config.run.tracking_uri)
            else:
                return tracker_class(config)
        except Exception as e:
            print(f"Error initializing tracker '{logger_type}': {e}")
            return None

def get_tracker(config, rank: int) -> Optional[ExperimentTracker]:
    """Factory function to get the appropriate experiment tracker."""
    return TrackerRegistry.get_tracker(config, rank)
