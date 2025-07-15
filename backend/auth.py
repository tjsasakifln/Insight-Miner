from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import re

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, status
import pyotp
import qrcode
from io import BytesIO
import base64

from .config import settings
from .logger import get_logger

logger = get_logger(__name__)

pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12  # Increased rounds for better security
)

class PasswordStrengthValidator:
    """Enhanced password strength validation."""
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """Validate password strength according to enterprise standards."""
        if len(password) < settings.security.password_min_length:
            return False, f"Password must be at least {settings.security.password_min_length} characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        if settings.security.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        # Check for common weak passwords
        common_passwords = {"password", "123456", "qwerty", "admin", "letmein"}
        if password.lower() in common_passwords:
            return False, "Password is too common"
        
        return True, "Password is strong"


class SecureAuthenticator:
    """Enterprise-grade authentication with security features."""
    
    def __init__(self):
        self.password_validator = PasswordStrengthValidator()
        self.failed_attempts = {}  # In production, use Redis
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password with timing attack protection."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Hash password with secure bcrypt."""
        # Validate password strength first
        is_valid, message = self.password_validator.validate_password_strength(password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return pwd_context.hash(password)
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if user is rate limited."""
        if identifier not in self.failed_attempts:
            return True
        
        attempts, last_attempt = self.failed_attempts[identifier]
        if attempts >= self.max_failed_attempts:
            if datetime.utcnow() - last_attempt < self.lockout_duration:
                return False
            else:
                # Reset after lockout period
                del self.failed_attempts[identifier]
        
        return True
    
    def record_failed_attempt(self, identifier: str):
        """Record failed authentication attempt."""
        if identifier in self.failed_attempts:
            attempts, _ = self.failed_attempts[identifier]
            self.failed_attempts[identifier] = (attempts + 1, datetime.utcnow())
        else:
            self.failed_attempts[identifier] = (1, datetime.utcnow())
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts after successful authentication."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create secure JWT access token."""
        to_encode = data.copy()
        
        # Add security claims
        to_encode.update({
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32),  # JWT ID for revocation
            "sub": data.get("sub", "unknown"),
            "aud": "insight-miner",
            "iss": "insight-miner-auth"
        })
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.security.jwt_expiration_hours)
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(
                to_encode, 
                settings.security.jwt_secret_key, 
                algorithm=settings.security.jwt_algorithm
            )
            logger.info(f"Access token created for user: {data.get('sub', 'unknown')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create secure refresh token."""
        data = {
            "sub": user_id,
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)
        }
        
        expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last 7 days
        data.update({"exp": expire})
        
        return jwt.encode(
            data,
            settings.security.jwt_secret_key,
            algorithm=settings.security.jwt_algorithm
        )
    
    def decode_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT access token."""
        try:
            payload = jwt.decode(
                token,
                settings.security.jwt_secret_key,
                algorithms=[settings.security.jwt_algorithm],
                audience="insight-miner",
                issuer="insight-miner-auth"
            )
            
            # Validate token type
            if payload.get("type") == "refresh":
                logger.warning("Refresh token used as access token")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token decoding error: {e}")
            return None
    
    def generate_2fa_secret(self, user_email: str) -> tuple[str, str]:
        """Generate 2FA secret and QR code."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Generate QR code
        provisioning_uri = totp.provisioning_uri(
            user_email,
            issuer_name="Insight Miner"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for web display
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return secret, img_str
    
    def verify_2fa_token(self, secret: str, token: str) -> bool:
        """Verify 2FA token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False


# Global authenticator instance
authenticator = SecureAuthenticator()

# Backward compatibility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return authenticator.verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return authenticator.get_password_hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    return authenticator.create_access_token(data, expires_delta)

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    return authenticator.decode_access_token(token)
