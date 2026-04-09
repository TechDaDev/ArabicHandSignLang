# Alembic

This folder contains database migration configuration for the Arabic Hand Sign Language backend.

## Common commands

```powershell
alembic revision --autogenerate -m "describe change"
alembic upgrade head
alembic downgrade -1
```

Alembic reads the active `DATABASE_URL` from `.env` through `app.core.config`.
