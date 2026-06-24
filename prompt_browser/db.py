"""Shared PostgreSQL connection helper."""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_dsn():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit(
            "DATABASE_URL is not set. Copy .env.example to .env and set it, "
            "e.g. postgresql://prompts:prompts@localhost:5432/prompts"
        )
    return dsn


def connect():
    return psycopg2.connect(get_dsn())
