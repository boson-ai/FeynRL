import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import mlflow
import wandb

def _flatten_dict(d, parent_key='', sep='.'):
    '''
        Flatten a nested dict into dot-separated keys.
    '''
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        '''
           Log hyperparameters of the experiment.
        '''
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        '''
           Log training or evaluation metrics.
        '''
        pass

    @abstractmethod
    def finish(self):
        '''
           Signal that the experiment run has ended.
        '''
        pass

class MLFlowTracker(ExperimentTracker):
    def __init__(self, config, tracking_uri: str):
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(config.run.project_name)
        self.run = self.mlflow.start_run(run_name=config.run.experiment_id)

        # Log all config parameters with section-prefixed dot-notation keys
        full_config = config.model_dump()
        flat_params = _flatten_dict(full_config)
        # Filter None values and convert non-scalar types to strings for mlflow
        params = {}
        for k, v in flat_params.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict)):
                v = str(v)
            params[k] = v
        self.log_params(params)

    def log_params(self, params: Dict[str, Any]):
        # MLflow has a batch size limit; log in chunks of 100
        items = list(params.items())
        for i in range(0, len(items), 100):
            self.mlflow.log_params(dict(items[i:i+100]))

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.mlflow.log_metrics(metrics, step=step)

    def finish(self):
        self.mlflow.end_run()

class WandBTracker(ExperimentTracker):
    def __init__(self, config):
        self.wandb = wandb

        # Load API key from file
        key_path = "./.wandb_key"
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = api_key

        # Log full config — wandb handles nested dicts natively,
        # so section-prefixed keys avoid collisions across sections.
        wandb_config = config.model_dump(exclude_none=True)

        self.run = self.wandb.init(
            project=config.run.project_name,
            name=config.run.experiment_id,
            config=wandb_config,
        )

        # Tell wandb to use "global_step" as the x-axis for all metrics.
        # This avoids the step= parameter which causes buffering and
        # merge/overwrite issues when multiple log() calls share a step.
        self.wandb.define_metric("global_step")
        self.wandb.define_metric("*", step_metric="global_step")

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

        if step is not None:
            formatted_metrics["global_step"] = step
        self.wandb.log(formatted_metrics)

    def finish(self):
        self.wandb.finish()

class TrackerRegistry:
    '''
        Registry for experiment trackers.
        Notes:
        - Only rank 0 will have a tracker.
        - Default tracker is mlflow.
    '''
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
    '''
        Factory function to get the appropriate experiment tracker.
    '''
    return TrackerRegistry.get_tracker(config, rank)
