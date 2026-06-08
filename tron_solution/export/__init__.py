"""Tron Export Package."""

__all__ = ["export_model"]


def __getattr__(name):
    if name == "export_model":
        from tron_solution.export.export_model import export_model
        return export_model
    raise AttributeError(name)
