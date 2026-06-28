"""Ekstensi Flask terpusat (hindari circular import)."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
