"""Tron Test Package."""

__all__ = ["test_model"]


def __getattr__(name):
    if name == "test_model":
        from tron_solution.test.test_model import test_model
        return test_model
    raise AttributeError(name)
