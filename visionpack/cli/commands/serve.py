from __future__ import annotations

import argparse

from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Serve the local web UI + REST API for this dataset",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=8123, help="Port (default: 8123)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the UI in a browser automatically",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    project = Project.open(".")
    try:
        import uvicorn

        from visionpack.server.app import create_app
    except ModuleNotFoundError as exc:
        raise VisionPackError(
            "`vp serve` needs the optional server backend. Install it with: pip install 'visionpack[server]'."
        ) from exc

    app = create_app(project.root)
    url = f"http://{args.host}:{args.port}"
    print(f"VisionPack UI for {project.manifest.name!r} at {url}  (API docs: {url}/api/docs)")
    if not args.no_browser:
        import webbrowser

        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
