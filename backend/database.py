from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "postgresql://user:password@db:5432/insight_miner"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user") # e.g., "user", "admin"

    uploads = relationship("UploadHistory", back_populates="uploader")
    analyses = relationship("AnalysisMetadata", back_populates="analyst")
    configurations = relationship("UserConfiguration", back_populates="user")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    details = Column(String)

class UploadHistory(Base):
    __tablename__ = "upload_history"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    uploader_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="uploaded") # e.g., "uploaded", "processing", "completed", "failed"

    uploader = relationship("User", back_populates="uploads")
    analysis_metadata = relationship("AnalysisMetadata", back_populates="upload_entry", uselist=False)

class AnalysisMetadata(Base):
    __tablename__ = "analysis_metadata"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("upload_history.id"))
    analysis_type = Column(String) # e.g., "sentiment", "topic"
    status = Column(String, default="pending") # e.g., "pending", "in_progress", "completed", "failed"
    result_summary = Column(Text, nullable=True)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    analyst_id = Column(Integer, ForeignKey("users.id"))

    upload_entry = relationship("UploadHistory", back_populates="analysis_metadata")
    analyst = relationship("User", back_populates="analyses")

class UserConfiguration(Base):
    __tablename__ = "user_configurations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    config_key = Column(String)
    config_value = Column(Text)

    user = relationship("User", back_populates="configurations")

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)