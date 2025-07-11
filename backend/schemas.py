from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    role: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UploadHistoryBase(BaseModel):
    file_name: str

class UploadHistoryCreate(UploadHistoryBase):
    pass

class UploadHistory(UploadHistoryBase):
    id: int
    upload_timestamp: datetime
    uploader_id: int
    status: str

    class Config:
        orm_mode = True

class AnalysisMetadataBase(BaseModel):
    upload_id: int
    analysis_type: str

class AnalysisMetadataCreate(AnalysisMetadataBase):
    pass

class AnalysisMetadata(AnalysisMetadataBase):
    id: int
    status: str
    result_summary: Optional[str]
    analysis_timestamp: datetime
    analyst_id: int

    class Config:
        orm_mode = True

class UserConfigurationBase(BaseModel):
    config_key: str
    config_value: str

class UserConfigurationCreate(UserConfigurationBase):
    pass

class UserConfiguration(UserConfigurationBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
