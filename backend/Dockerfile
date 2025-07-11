FROM python:3.9-slim-buster as builder
WORKDIR /app
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.9/site-packages/uvicorn /usr/local/lib/python3.9/site-packages/uvicorn
RUN pip install --no-cache-dir /wheels/*

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
