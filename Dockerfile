FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md openenv.yaml inference.py MANIFEST.in $HOME/app/
COPY --chown=user cybersoc_openenv $HOME/app/cybersoc_openenv
COPY --chown=user server $HOME/app/server

RUN python -m pip install .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).read()"

CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "7860"]
