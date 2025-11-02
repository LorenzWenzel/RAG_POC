# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Systempakete minimal halten (build-essential oft nötig für Wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Deps layer (besseres Caching)
FROM base AS deps
# COPY pyproject.toml poetry.lock ./
# RUN pip install --upgrade pip && pip install .
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Runtime
FROM base AS runtime
# Non-root user
RUN groupadd -r app && useradd -r -g app app

# Deps aus vorherigem Stage
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# App-Code
COPY . /app

# Streamlit Headless
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Port
EXPOSE 8502

# Rechte & User
RUN chown -R app:app /app
USER app

# Start (Passe Home.py an, falls nötig)
CMD ["streamlit", "run", "Home.py", "--server.port=8502", "--server.address=0.0.0.0"]
