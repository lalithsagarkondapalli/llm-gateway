"""Run with: python -m gateway  (reads GATEWAY_CONFIG, default config.yaml)."""
import logging
import os

import uvicorn

from .app import create_app

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")),
                log_level=os.environ.get("LOG_LEVEL", "info").lower())
