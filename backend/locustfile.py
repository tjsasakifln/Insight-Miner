from locust import HttpUser, task, between
import random

class InsightMinerUser(HttpUser):
    wait_time = between(1, 2)
    host = "http://backend:8000"
    token = None

    def on_start(self):
        self.client.verify = False # Disable SSL verification for local testing
        self.register_and_login()

    def register_and_login(self):
        username = f"testuser_{random.randint(0, 100000)}"
        email = f"{username}@example.com"
        password = "testpassword"

        # Register
        self.client.post("/register", json={
            "username": username,
            "email": email,
            "password": password
        })

        # Login
        response = self.client.post("/token", data={
            "username": username,
            "password": password
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
        else:
            print(f"Login failed for {username}: {response.text}")

    @task(3)
    def upload_file(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            files = {'file': ('mock_review.txt', 'This is a mock review content.', 'text/plain')}
            self.client.post("/upload", files=files, headers=headers)

    @task(1)
    def analyze_sentiment(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            text_to_analyze = "This is a sample text for sentiment analysis."
            self.client.post("/analyze_sentiment", params={"text": text_to_analyze}, headers=headers)
