import streamlit as st
import requests

BACKEND_URL = "http://backend:8000"

st.title("Insight Miner - Frontend")

# User Authentication
st.sidebar.header("Authentication")

if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.token is None:
    auth_option = st.sidebar.radio("Choose an option", ("Login", "Register"))

    if auth_option == "Register":
        st.sidebar.subheader("Register")
        new_username = st.sidebar.text_input("New Username")
        new_email = st.sidebar.text_input("New Email")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            try:
                response = requests.post(f"{BACKEND_URL}/register", json={
                    "username": new_username,
                    "email": new_email,
                    "password": new_password
                })
                if response.status_code == 200:
                    st.sidebar.success("Registration successful! Please login.")
                else:
                    st.sidebar.error(f"Registration failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Could not connect to backend.")

    elif auth_option == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            try:
                response = requests.post(f"{BACKEND_URL}/token", data={
                    "username": username,
                    "password": password
                })
                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.token = token_data["access_token"]
                    st.sidebar.success("Logged in successfully!")
                    st.experimental_rerun()
                else:
                    st.sidebar.error(f"Login failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Could not connect to backend.")
else:
    st.sidebar.success("You are logged in.")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.experimental_rerun()

# Main content
st.write("This is the Streamlit frontend consuming the FastAPI backend.")

if st.button("Test Backend Connection"):
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            st.success(f"Backend connected successfully: {response.json()}")
        else:
            st.error(f"Failed to connect to backend: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Make sure the backend service is running.")

st.header("API Endpoints (Requires Login)")

headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

if st.session_state.token:
    if st.button("Upload File (Protected)"):
        try:
            response = requests.post(f"{BACKEND_URL}/upload", headers=headers)
            st.write(response.json())
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend.")

    if st.button("Analyze Sentiment (Admin Only)"):
        try:
            response = requests.post(f"{BACKEND_URL}/analyze_sentiment", headers=headers)
            st.write(response.json())
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend.")

    if st.button("Extract Topics (Protected)"):
        try:
            response = requests.post(f"{BACKEND_URL}/extract_topics", headers=headers)
            st.write(response.json())
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend.")

    if st.button("Generate Report (Admin Only)"):
        try:
            response = requests.post(f"{BACKEND_URL}/generate_report", headers=headers)
            st.write(response.json())
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend.")
else:
    st.info("Please login to access protected API endpoints.")