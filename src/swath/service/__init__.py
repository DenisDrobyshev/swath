"""The web service: a small FastAPI application around a trained checkpoint."""

from swath.service.app import create_app

__all__ = ["create_app"]
