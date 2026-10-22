# syntax=docker/dockerfile:1
# Lang Reader: web app and generation worker share this image (see compose.yaml).

FROM node:22-slim AS web
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.json ./
RUN pnpm install --frozen-lockfile
COPY web ./web
RUN pnpm build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock uv.toml README.md LICENSE NOTICE ./
RUN uv sync --locked --no-dev --no-install-project
COPY server ./server
COPY packs ./packs
COPY examples ./examples
COPY scripts ./scripts
COPY --from=web /app/server/static ./server/static
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH" ARC_LANG_DATA_DIR=/data
RUN useradd --system --create-home lang && mkdir /data && chown lang /data
USER lang
VOLUME /data
EXPOSE 8000
# Binds all interfaces inside the container; compose publishes it on the host's loopback only.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
