from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from typing import Annotated
import os
from dotenv import load_dotenv
import redis.asyncio as redis
import uuid
from prometheus_client import generate_latest, Counter, Histogram
import pandas as pd
import io

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_ipaddr
from slowapi.errors import RateLimitExceeded

from .database import SessionLocal, engine, Base, User, create_db_and_tables, AuditLog, UploadHistory
from .auth import get_password_hash, verify_password, create_access_token, decode_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from .sentiment_analysis import sentiment_analyzer
from .schemas import UserCreate, User, Token, UploadHistory as UploadHistorySchema, AnalysisMetadataCreate
from .celery_worker import celery_app, process_file_task
from .openai_integration import openai_integration
from loguru import logger

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

load_dotenv() # Load environment variables from .env file

# Configure OpenTelemetry
resource = Resource.create({"service.name": "insight-miner-backend"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_AGENT_HOST", "jaeger"),
    agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
)
span_processor = BatchSpanProcessor(jaeger_exporter)
tracer_provider.add_span_processor(span_processor)

# Instrument FastAPI
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

# Instrument SQLAlchemy
SQLAlchemyInstrumentor().instrument(engine=engine)

limiter = Limiter(key_func=get_ipaddr, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])
UPLOAD_COUNT = Counter('file_uploads_total', 'Total File Uploads')
SENTIMENT_ANALYSIS_COUNT = Counter('sentiment_analysis_total', 'Total Sentiment Analysis Requests')
ERROR_COUNT = Counter('http_errors_total', 'Total HTTP Errors', ['method', 'endpoint', 'status_code'])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# OAuth2PasswordBearer for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Function to get current user from token
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Function to check user role for RBAC
def check_role(required_role: str):
    def role_checker(current_user: Annotated[User, Depends(get_current_user)]):
        if current_user.role != required_role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
        return current_user
    return role_checker

# Audit logging function
def log_audit_event(db: Session, user_id: int, action: str, details: str = None):
    audit_log = AuditLog(timestamp=datetime.utcnow(), user_id=user_id, action=action, details=details)
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)
    logger.info(f"Audit: User {user_id} - {action} - {details or ''}")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
@limiter.limit("5/second")
async def read_root(request: Request):
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to Insight Miner Backend!"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health/live")
async def health_live():
    logger.info("Liveness check performed.")
    return {"status": "ok"}

@app.get("/health/ready")
async def health_ready(db: Session = Depends(get_db)):
    try:
        # Check PostgreSQL connection
        db.execute("SELECT 1")

        # Check Redis connection
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
        await r.ping()

        logger.info("Readiness check successful.")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service not ready: {e}")

@app.post("/register", response_model=User)
@limiter.limit("5/minute")
async def register_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    log_audit_event(db, db_user.id, "user_registration", f"New user registered: {user.username}")
    logger.info(f"User {user.username} registered successfully.")
    return db_user

@app.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    log_audit_event(db, user.id, "user_login", f"User logged in: {user.username}")
    logger.info(f"User {user.username} logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload", response_model=UploadHistorySchema)
@limiter.limit("5/minute")
async def upload_file(request: Request, file: UploadFile = File(...), current_user: Annotated[User, Depends(get_current_user)], db: Session = Depends(get_db)):
    UPLOAD_COUNT.inc()
    # Read the uploaded file content
    contents = await file.read()
    file_stream = io.StringIO(contents.decode('utf-8'))

    # Validate CSV format and required columns
    try:
        df = pd.read_csv(file_stream)
        if "review_text" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'review_text' column.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file or format: {e}")

    # Save the file temporarily
    file_location = f"./uploaded_files/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb+") as file_object:
        file_object.write(contents)

    # Record upload history
    upload_entry = UploadHistory(file_name=file.filename, uploader_id=current_user.id, status="processing")
    db.add(upload_entry)
    db.commit()
    db.refresh(upload_entry)

    # Dispatch to Celery for async processing
    process_file_task.delay(file_location, upload_entry.id, current_user.email)

    log_audit_event(db, current_user.id, "file_upload", f"User {current_user.username} uploaded file {file.filename}. Task dispatched.")
    logger.info(f"File {file.filename} uploaded by user {current_user.username}. Task dispatched to Celery.")
    return upload_entry

@app.post("/analyze_sentiment")
@limiter.limit("10/minute")
async def analyze_sentiment(request: Request, text: str, current_user: Annotated[User, Depends(check_role("admin"))], db: Session = Depends(get_db)):
    SENTIMENT_ANALYSIS_COUNT.inc()
    log_audit_event(db, current_user.id, "sentiment_analysis", f"Admin {current_user.username} performed sentiment analysis on: {text}")
    try:
        result = await sentiment_analyzer.analyze_sentiment(text)
        logger.info(f"Sentiment analysis completed for text: {text}")
        return {"message": "Sentiment analysis completed", "result": result, "user": current_user.username}
    except HTTPException as e:
        logger.error(f"Sentiment analysis failed for text: {text} - {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Sentiment analysis failed for text: {text} - {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

@app.post("/extract_topics")
@limiter.limit("10/minute")
async def extract_topics(request: Request, current_user: Annotated[User, Depends(get_current_user)], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "topic_extraction", f"User {current_user.username} extracted topics.")
    logger.info(f"Topic extraction requested by user {current_user.username}.")
    return {"message": "Topic extraction endpoint (not yet implemented)", "user": current_user.username}

@app.post("/generate_report")
@limiter.limit("2/minute")
async def generate_report(request: Request, current_user: Annotated[User, Depends(check_role("admin"))], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "report_generation", f"Admin {current_user.username} generated a report.")
    logger.info(f"Report generation requested by admin {current_user.username}.")
    return {"message": "Report generation endpoint (not yet implemented)", "user": current_user.username}

@app.post("/summarize")
@limiter.limit("5/minute")
async def summarize_text(request: Request, text: str, current_user: Annotated[User, Depends(get_current_user)]):
    log_audit_event(db, current_user.id, "text_summarization", f"User {current_user.username} requested summary for text: {text[:50]}...")
    try:
        summary = await openai_integration.get_chat_completion(f"Summarize the following text: {text}")
        logger.info(f"Text summarization completed for text: {text[:50]}...")
        return {"message": "Text summarization completed", "summary": summary, "user": current_user.username}
    except Exception as e:
        logger.error(f"Text summarization failed for text: {text[:50]}... - {e}")
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text() # Keep connection alive
            # You can add logic here to handle messages from the client if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")