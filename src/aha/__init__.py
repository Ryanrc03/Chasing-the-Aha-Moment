"""GRPO post-training experiments for "Chasing the Aha Moment".

Heavy dependencies (unsloth, trl, vllm, datasets) are imported lazily inside
functions so that this package, its configs and its unit tests load on any
machine, including Windows without a GPU.
"""

from aha.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
