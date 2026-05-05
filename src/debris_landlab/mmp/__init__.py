"""Multi-model landslide probability workflow."""

from debris_landlab.mmp.config import MMPConfig, load_mmp_config

__all__ = ["MMPConfig", "load_mmp_config", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from debris_landlab.mmp.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(name)
