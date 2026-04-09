# Alembic Migrations

This folder is reserved for database migrations.

## Initialize or refresh migrations

```bash
alembic revision --autogenerate -m "create initial tables"
alembic upgrade head
```

Use `DATABASE_URL` from `.env` to target PostgreSQL in development or production.
