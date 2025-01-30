_MAJOR = "0"
_MINOR = "1"
# Keep patch one ahead on main; add suffix for nightly (e.g., .devYYYYMMDD)
_PATCH = "0"
_SUFFIX = ""

VERSION_SHORT = f"{_MAJOR}.{_MINOR}"
VERSION = f"{_MAJOR}.{_MINOR}.{_PATCH}{_SUFFIX}"

__all__ = ["VERSION", "VERSION_SHORT"]
