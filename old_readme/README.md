# Alpha Research Platform

Professional-grade investment research platform combining quantitative screening with AI-powered fundamental analysis.

## Features

- **Quantitative Screener**: Scores 100+ stocks on sentiment, fundamentals, technicals, and analyst data
- **AI Committee**: Multi-agent debate system for deep-dive analysis with SEC filing RAG
- **Real-time Control**: Pause/resume pipelines to save energy
- **Kafka Messaging**: Decoupled architecture for scalability
- **Streamlit Dashboard**: Professional visualization and control panel

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Alpha Research Platform                       │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard (Control + Visualization)                   │
├─────────────────────────────────────────────────────────────────┤
│  Kafka (Redpanda) - Message Bus                                  │
├──────────────────────┬──────────────────────────────────────────┤
│  Screener Worker     │  Committee Worker                         │
│  (100 tickers)       │  (Deep analysis)                          │
├──────────────────────┴──────────────────────────────────────────┤
│  TimescaleDB + pgvector (Data Layer)                            │
├─────────────────────────────────────────────────────────────────┤
│  Local LLMs: Qwen-Next-80B (8080) + GPT-OSS-20B (8081)          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Local LLMs running (Qwen on 8080, GPT-OSS on 8081)

### 1. Clone and Setup

```bash
cd alpha-platform

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Infrastructure

```bash
# Start TimescaleDB and Kafka
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
# Database schema is auto-created by Docker
# Verify tables exist:
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "\dt"
```

### 5. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
alpha-platform/
├── config/
│   ├── settings.yaml      # Main configuration
│   └── universe.csv       # 100 tickers to track
├── sql/
│   └── 001_schema.sql     # Database schema
├── src/
│   ├── db/                # Database access layer
│   ├── kafka/             # Kafka producer/consumer
│   ├── data/              # Data ingestion
│   ├── screener/          # Quantitative screening
│   ├── committee/         # AI agent system
│   ├── llm/               # LLM providers
│   ├── portfolio/         # Portfolio management
│   └── utils/             # Utilities
├── dashboard/
│   ├── app.py             # Streamlit main
│   ├── pages/             # Dashboard pages
│   └── components/        # Reusable components
├── workers/               # Background workers
├── docker-compose.yml     # Infrastructure
└── requirements.txt       # Python dependencies
```

## Configuration

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | Database password | (required) |
| `LLM_QWEN_BASE_URL` | Qwen model endpoint | `http://172.23.193.91:8080/v1` |
| `LLM_GPT_OSS_BASE_URL` | GPT-OSS endpoint | `http://172.23.193.91:8081/v1` |
| `FINNHUB_API_KEY` | Finnhub API key | (optional) |

### Screener Weights (settings.yaml)

```yaml
screener:
  weights:
    sentiment: 0.15
    fundamental: 0.30
    growth: 0.20
    dividend: 0.05
    gap: 0.05
    analyst: 0.05
    target: 0.10
    likelihood: 0.10
```

## Usage

### Energy Management

```bash
# Pause all containers (saves energy)
docker-compose pause

# Resume
docker-compose unpause

# Stop completely
docker-compose down
```

### Kafka Console

Access Redpanda Console at: http://localhost:8080

### Database Access

```bash
# Connect to database
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform

# Example queries
SELECT * FROM v_latest_scores ORDER BY total_score DESC LIMIT 10;
SELECT * FROM trading_signals WHERE date = CURRENT_DATE;
```

## Development Status

- [x] Phase 1: Infrastructure (TimescaleDB, Kafka, Schema)
- [ ] Phase 2: Data Migration (Project 1 → TimescaleDB)
- [ ] Phase 3: Analysis Integration (Screener + Committee)
- [ ] Phase 4: Dashboard & Polish

## License

Proprietary - Internal Use Only
