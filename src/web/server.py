"""Starts the local web UI with uvicorn."""

from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI

from src.config.settings import Settings, get_settings
from src.web.app import create_app, set_shutdown_hook


def port_is_free(host: str, port: int) -> bool:
    """Check the port up front - uvicorn exits the process on bind errors itself."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def local_url(host: str, port: int) -> str:
    hostname = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
    return f"http://{hostname}:{port}"


def run_server(
    settings: Settings | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    log_level: str = "warning",
    app: FastAPI | None = None,
) -> None:
    if not port_is_free(host, port):
        raise OSError(f"Port {port} on {host} is already in use.")

    settings = settings or get_settings()
    app = app or create_app(settings)

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level.lower())
    server = uvicorn.Server(config)

    # Lets the Quit button in the UI stop the server cleanly.
    set_shutdown_hook(app, lambda: setattr(server, "should_exit", True))

    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(local_url(host, port),)).start()

    server.run()

    # uvicorn swallows bind errors itself; ``started`` then stays False.
    if not server.started:
        raise OSError(f"Port {port} on {host} could not be opened.")


def main() -> None:
    """Entry point for ``python -m src.web.server``."""
    run_server()


if __name__ == "__main__":
    main()
