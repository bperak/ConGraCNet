"""
Auto-load .env and override legacy authSettings module at import time.

This allows containerized deployments to configure credentials via environment
variables without modifying application code.

Supported variables:
- GRAPH_URL
- GRAPH_USER
- GRAPH_PASS
- SKETCH_USER
- SKETCH_API_KEY
"""

import os

try:
    from dotenv import load_dotenv

    # Load .env if present
    load_dotenv()
except Exception:
    pass


def _maybe_override_authsettings() -> None:
    """Override attributes in authSettings if env vars are provided."""

    try:
        import authSettings  # type: ignore
    except Exception:
        return

    mapping = {
        "graphURL": os.getenv("GRAPH_URL"),
        "graphUser": os.getenv("GRAPH_USER"),
        "graphPass": os.getenv("GRAPH_PASS"),
        "userName": os.getenv("SKETCH_USER"),
        "apiKey": os.getenv("SKETCH_API_KEY"),
    }

    for attribute_name, env_value in mapping.items():
        if env_value:
            setattr(authSettings, attribute_name, env_value)


_maybe_override_authsettings()


