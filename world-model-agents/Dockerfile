# MIKO world-model agents: live HTTP service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MIKO_DATA_DIR=/data MIKO_PROVIDER=yahoo MIKO_REFRESH_SECONDS=1800 MIKO_DATA_CACHE=/app/data_cache
WORKDIR /app

COPY pyproject.toml README.md ./
COPY miko ./miko
RUN pip install --no-cache-dir ".[live]"

# bundled public research data (used by the replay provider and the scenario-replay windows)
COPY data_cache ./data_cache

RUN useradd --create-home miko && mkdir -p /data && chown miko /data
USER miko
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
# one worker: the models live in memory and refresh in a background thread
CMD ["uvicorn", "miko.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
