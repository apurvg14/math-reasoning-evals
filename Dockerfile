FROM python:3.12-slim

WORKDIR /app

# Real-model SDKs (the core harness itself is stdlib-only).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mathrobust ./mathrobust

EXPOSE 8765

# Bind all interfaces so the dashboard is reachable from the host.
CMD ["python", "-m", "mathrobust", "dashboard", "--host", "0.0.0.0", "--no-open"]
