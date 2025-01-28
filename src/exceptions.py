class ElevationError(Exception):
    """Base exception for Elevation."""


class ConfigurationError(ElevationError):
    """Raised for invalid configuration values or incompatible settings."""


class CheckpointError(ElevationError):
    """Raised when loading/saving model state fails or is incompatible."""


class TokenizerError(ElevationError):
    """Raised for tokenizer initialization or I/O failures."""

__all__ = [
    "ElevationError",
    "ConfigurationError",
    "CheckpointError",
    "TokenizerError",
]
