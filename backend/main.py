from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Annotated

from .database import SessionLocal, engine, Base, User, create_db_and_tables, AuditLog
from .auth import get_password_hash, verify_password, create_access_token, decode_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from .sentiment_analysis import sentiment_analyzer
from loguru import logger

app = FastAPI()

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
    audit_log = AuditLog(timestamp=str(datetime.utcnow()), user_id=user_id, action=action, details=details)
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)
    logger.info(f"Audit: User {user_id} - {action} - {details or ''}")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Insight Miner Backend!"}

@app.post("/register")
async def register_user(username: str, email: str, password: str, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    log_audit_event(db, db_user.id, "user_registration", f"New user registered: {username}")
    return {"message": "User registered successfully"}

@app.post("/token")
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
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
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_file(current_user: Annotated[User, Depends(get_current_user)], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "file_upload", f"User {current_user.username} uploaded a file.")
    return {"message": "File upload endpoint (not yet implemented)", "user": current_user.username}

@app.post("/analyze_sentiment")
async def analyze_sentiment(text: str, current_user: Annotated[User, Depends(check_role("admin"))], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "sentiment_analysis", f"Admin {current_user.username} performed sentiment analysis on: {text}")
    try:
        result = await sentiment_analyzer.analyze_sentiment(text)
        return {"message": "Sentiment analysis completed", "result": result, "user": current_user.username}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

@app.post("/extract_topics")
async def extract_topics(current_user: Annotated[User, Depends(get_current_user)], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "topic_extraction", f"User {current_user.username} extracted topics.")
    return {"message": "Topic extraction endpoint (not yet implemented)", "user": current_user.username}

@app.post("/generate_report")
async def generate_report(current_user: Annotated[User, Depends(check_role("admin"))], db: Session = Depends(get_db)):
    log_audit_event(db, current_user.id, "report_generation", f"Admin {current_user.username} generated a report.")
    return {"message": "Report generation endpoint (not yet implemented)", "user": current_user.username}