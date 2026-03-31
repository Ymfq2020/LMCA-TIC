"""Optional dependency guards."""

from __future__ import annotations


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


def require_dependency(module: object | None, package_name: str) -> None:
    if module is None:
        raise MissingDependencyError(
            f"Missing optional dependency '{package_name}'. "
            f"Install the package or use the smoke configuration."
        )
