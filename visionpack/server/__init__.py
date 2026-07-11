"""Local API server + web UI (`vp serve`).

FastAPI/uvicorn are optional (`pip install "visionpack[server]"`); nothing in
this package is imported by the core, and `visionpack.server.app` imports the
web stack only when actually used.
"""
