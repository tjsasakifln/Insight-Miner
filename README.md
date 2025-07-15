# 🚀 Insight Miner - Enterprise Customer Feedback Analysis Platform

> **Transforming Customer Feedback into Actionable Business Intelligence with AI-Powered Sentiment Analysis**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/tjsasakifln/Insight-Miner)

**🌍 Open for International Opportunities & Remote Collaboration**

</div>

## 🌟 Overview

**Insight Miner** is an enterprise-grade customer feedback analysis platform built with **Clean Architecture** and **Domain-Driven Design (DDD)** principles. It transforms raw customer feedback into actionable business insights through advanced sentiment analysis, topic extraction, and AI-powered summarization.

### 🎯 Business Problem Solved

Companies struggle to manually analyze thousands of customer reviews and feedback data. Product and marketing teams miss opportunities to quickly understand customer sentiment, identify pain points, and extract actionable insights from unstructured feedback data.

### 💡 Solution Approach

Insight Miner automates the entire customer feedback analysis pipeline using enterprise-grade architecture, multi-provider AI services, and real-time processing capabilities to deliver immediate business value.

### 🎯 Key Features

- **🔄 Real-time Processing** - Asynchronous file processing with live progress updates
- **🤖 Multi-Provider AI** - Google Cloud NLP, AWS Comprehend, and OpenAI integration
- **📊 Advanced Analytics** - Sentiment trends, anomaly detection, and business metrics
- **🔐 Enterprise Security** - JWT authentication, 2FA, rate limiting, and audit logging
- **⚡ High Performance** - Batch processing, circuit breakers, and optimized caching
- **📱 Modern UI** - Interactive dashboards with real-time WebSocket updates
- **🔧 Microservices Ready** - Scalable architecture with containerization support

## 🏗️ Architecture

### Enterprise-Grade Clean Architecture

```
┌─────────────────────────────────────────────────┐
│                   API Layer                     │
│              (FastAPI + WebSocket)              │
├─────────────────────────────────────────────────┤
│               Application Layer                 │
│           (Use Cases + Services)                │
├─────────────────────────────────────────────────┤
│                Domain Layer                     │
│    (Entities + Value Objects + Events)         │
├─────────────────────────────────────────────────┤
│              Infrastructure Layer               │
│  (Database + Redis + External APIs + Queue)    │
└─────────────────────────────────────────────────┘
```

### 🔧 Technology Stack

#### Core Framework
- **FastAPI** - High-performance async web framework
- **Pydantic** - Data validation and settings management
- **SQLAlchemy** - ORM with async support
- **PostgreSQL** - Primary database with advanced features

#### Processing & Queue
- **Celery** - Distributed task queue for background processing
- **Redis** - In-memory caching and message broker
- **AsyncIO** - High-performance async operations

#### AI & ML
- **OpenAI GPT-4** - Advanced text analysis and summarization
- **Google Cloud NLP** - Sentiment analysis and entity extraction
- **AWS Comprehend** - Large-scale text analysis

#### Monitoring & Observability
- **Prometheus** - Metrics collection and alerting
- **Grafana** - Visualization and dashboards
- **Jaeger** - Distributed tracing
- **Structured Logging** - JSON-based logging with correlation IDs

## 📦 Installation & Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 14+
- Redis 7+

### 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tjsasakifln/Insight-Miner.git
   cd Insight-Miner
   ```

2. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Configure your API keys and database settings
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Start the application:**
   ```bash
   # Backend API
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   
   # Celery Worker (separate terminal)
   celery -A backend.celery_worker worker --loglevel=info
   
   # Frontend (separate terminal)
   streamlit run frontend/streamlit_app.py
   ```

### 🐳 Production Deployment

Production deployment requires proper database setup (PostgreSQL), Redis instance, and environment configuration. The application is designed for containerized deployment with Docker and Kubernetes support.

## 🎯 Usage

### API Endpoints

#### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User authentication
- `POST /auth/refresh` - Refresh JWT token

#### File Management
- `POST /files/upload` - Upload CSV file for analysis
- `GET /files/{upload_id}/status` - Check processing status
- `GET /files/{upload_id}/results` - Get analysis results

#### Analytics
- `GET /analytics/sentiment/{upload_id}` - Sentiment analysis results
- `GET /analytics/dashboard` - Analytics dashboard data
- `GET /analytics/export/{upload_id}` - Export analysis results

#### System
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation

### 📊 Dashboard Features

- **Real-time Progress** - Live updates during file processing
- **Sentiment Visualization** - Interactive charts and graphs
- **Topic Analysis** - Key themes and discussion points
- **Trend Analysis** - Historical sentiment trends
- **Anomaly Detection** - Statistical outliers identification
- **Export Capabilities** - CSV, JSON, and PDF reports

## 🔐 Security Features

### Authentication & Authorization
- **JWT tokens** with secure secret management
- **Two-factor authentication** (2FA) support
- **Role-based access control** (RBAC)
- **Rate limiting** to prevent abuse
- **Session management** with Redis

### Data Protection
- **Input validation** and sanitization
- **SQL injection** prevention
- **XSS protection** with CSP headers
- **Audit logging** for compliance
- **Encryption** for sensitive data

## 📈 Performance & Scalability

### Performance Optimizations
- **Async processing** with proper sync/async bridges
- **Batch operations** for database efficiency
- **Connection pooling** for optimal resource usage
- **Intelligent caching** with TTL management
- **Circuit breakers** for external service resilience

### Scalability Features
- **Horizontal scaling** with load balancing
- **Microservices architecture** for independent scaling
- **Database sharding** support
- **CDN integration** for static assets
- **Auto-scaling** with Kubernetes HPA

## 🔧 Configuration

### Environment Variables

```bash
# Security
SECURITY_JWT_SECRET_KEY=your-super-secure-jwt-secret
SECURITY_ENCRYPTION_KEY=your-encryption-key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=insight_miner
DB_USERNAME=postgres
DB_PASSWORD=your-password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# External Services
EXTERNAL_OPENAI_API_KEY=sk-your-openai-key
EXTERNAL_GOOGLE_CREDENTIALS_PATH=/path/to/credentials.json
EXTERNAL_AWS_ACCESS_KEY_ID=your-aws-key
EXTERNAL_AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### Application Settings

```python
# Application configuration
APP_ENVIRONMENT=production
APP_DEBUG=false
APP_LOG_LEVEL=INFO
APP_WORKERS=4
APP_MAX_CONCURRENT_TASKS=10
```

## 📊 Monitoring & Observability

### Metrics
- **Application metrics** - Request rates, response times, error rates
- **Business metrics** - Upload success rates, processing times
- **Infrastructure metrics** - CPU, memory, disk usage
- **External service metrics** - API response times, success rates

### Logging
- **Structured logging** with JSON format
- **Correlation IDs** for request tracing
- **Security events** logging
- **Audit trails** for compliance
- **Log aggregation** with ELK stack support

### Alerting
- **Prometheus alerts** for system issues
- **Business alerts** for processing failures
- **Security alerts** for suspicious activity
- **Capacity alerts** for resource planning

## 🧪 Testing

The application includes comprehensive testing strategies:

- **Unit tests** for domain logic validation
- **Integration tests** for API endpoint verification
- **Performance tests** for scalability assessment
- **Security tests** for vulnerability scanning

## 📚 Documentation

### API Documentation
- **OpenAPI/Swagger** - Interactive API documentation available at `/docs` when running the application
- **ReDoc** - Alternative documentation format at `/redoc`

### Architecture Documentation
- **Clean Architecture** implementation with Domain-Driven Design
- **Enterprise patterns** and best practices
- **Security guidelines** and compliance features

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/new-feature`
3. **Follow coding standards:** Use `black`, `isort`, `flake8`
4. **Write tests** for new features
5. **Update documentation** as needed
6. **Submit pull request**

### Code Quality Standards

- **Type hints** required for all functions
- **Docstrings** for all public methods
- **Comprehensive test coverage**
- **Security scanning** with `bandit`
- **Dependency scanning** with `safety`

## 🌍 International Opportunities & Global Collaboration

**I am actively seeking international opportunities and global collaboration projects.**

### What I Offer:
- **Enterprise-grade software architecture** and development
- **Clean Architecture & Domain-Driven Design** expertise
- **AI/ML integration** and advanced analytics solutions
- **Performance optimization** and scalability engineering
- **Security-first development** practices
- **Team leadership** and mentoring

### Available For:
- **Full-time positions** (international, remote, or on-site)
- **Consulting projects** and technical advisory
- **Architecture review** and system optimization
- **Team training** and knowledge transfer
- **Open source collaboration** and contributions

### Contact Information:
- **Email:** [tiago@confenge.com.br](mailto:tiago@confenge.com.br)
- **GitHub:** [github.com/tjsasakifln](https://github.com/tjsasakifln)
- **LinkedIn:** [linkedin.com/in/tiagosasaki](https://www.linkedin.com/in/tiagosasaki)
- **Location:** Brazil (Open to global opportunities)

---

<div align="center">

**Transform your customer feedback into actionable business insights with Insight Miner.**

Built with ❤️ by [Tiago Sasaki](mailto:tiago@confenge.com.br) | [**💬 Get in Touch**](mailto:tiago@confenge.com.br) | [**🔗 GitHub Issues**](https://github.com/tjsasakifln/Insight-Miner/issues)

</div>