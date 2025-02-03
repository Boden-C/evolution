class ModelError(Exception):
    """Base exception for Elevation."""


class ConfigurationError(ModelError):
    """Raised for invalid configuration values or incompatible settings."""


class CheckpointError(ModelError):
    """Raised when loading/saving model state fails or is incompatible."""


class TokenizerError(ModelError):
    """Raised for tokenizer initialization or I/O failures."""

__all__ = [
    "ModelError",
    "ConfigurationError",
    "CheckpointError",
    "TokenizerError",
]
