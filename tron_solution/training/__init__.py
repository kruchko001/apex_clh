"""Tron Training Package."""

__all__ = ["train"]

def __getattr__(name):
    if name == "train":
        from tron_solution.training.train_ppo import train
        return train
    raise AttributeError(name)
