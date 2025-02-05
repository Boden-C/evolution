import os
import sys
import logging
import socket
from .exceptions import CliError

def set_env_vars() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def install_excepthook() -> None:
    def excepthook(exc_type, value, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, value, tb)
        elif issubclass(exc_type, CliError):
            print(f"CLI error: {value}")
        else:
            print(f"Unhandled exception: {value}")
    sys.excepthook = excepthook

def filter_warnings() -> None:
    import warnings
    patterns = ["torch.distributed.*", "TypedStorage is deprecated.*", "failed to load.*"]
    for pat in patterns:
        warnings.filterwarnings("ignore", pat)

def prepare_cli_env() -> None:
    set_env_vars()
    install_excepthook()
    filter_warnings()
