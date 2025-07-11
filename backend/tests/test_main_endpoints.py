from unittest.mock import patch
from ..schemas import UserCreate

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Insight Miner Backend!"}

def test_register_user(client, db_session):
    user_data = UserCreate(username="testuser", email="test@example.com", password="testpassword")
    response = client.post("/register", json=user_data.dict())
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    assert response.json()["email"] == "test@example.com"

def test_login_for_access_token(client, db_session):
    # Register a user first
    user_data = UserCreate(username="loginuser", email="login@example.com", password="loginpassword")
    client.post("/register", json=user_data.dict())

    response = client.post("/token", data={"username": "loginuser", "password": "loginpassword"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_upload_file_unauthorized(client):
    response = client.post("/upload", files={"file": ("test.txt", "some content")})
    assert response.status_code == 401

def test_analyze_sentiment_unauthorized(client):
    response = client.post("/analyze_sentiment", params={"text": "test"})
    assert response.status_code == 401

def test_analyze_sentiment_admin_only(client, db_session):
    # Register an admin user
    user_data = UserCreate(username="adminuser", email="admin@example.com", password="adminpassword")
    client.post("/register", json=user_data.dict())
    admin_user = db_session.query(User).filter(User.username == "adminuser").first()
    admin_user.role = "admin"
    db_session.commit()

    # Login as admin
    token_response = client.post("/token", data={"username": "adminuser", "password": "adminpassword"})
    token = token_response.json()["access_token"]

    with patch('backend.sentiment_analysis.sentiment_analyzer.analyze_sentiment') as mock_analyze_sentiment:
        mock_analyze_sentiment.return_value = {"score": 0.5, "magnitude": 0.5, "provider": "mock"}
        response = client.post("/analyze_sentiment", params={"text": "test"}, headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        assert response.json()["result"]["score"] == 0.5

def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_ready(client):
    # This test assumes DB and Redis are accessible from the test environment
    # In a real scenario, you might mock these dependencies as well
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
