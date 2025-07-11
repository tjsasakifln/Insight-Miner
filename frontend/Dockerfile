FROM python:3.9-slim-buster as builder
WORKDIR /app
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.9/site-packages/streamlit /usr/local/lib/python3.9/site-packages/streamlit
RUN pip install --no-cache-dir /wheels/*

COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]