# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# Install uv by copying from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Compile bytecodes for faster startup and copy files instead of symlinking
# (required when cache and target are on separate filesystems).
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Install system libraries required by opencv-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies as a separate layer to take advantage of Docker's caching.
# Bind-mount pyproject.toml and uv.lock so they are available without being copied,
# then cache the uv download cache across builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the source code into the container.
COPY --chown=appuser:appuser . .

# Sync the project itself now that the source is present.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place the virtual environment's binaries on the PATH so the app can be run
# without explicitly calling `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]
