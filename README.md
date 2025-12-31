# 🍁 Made in Canada

A chat-based shopping experience for Canadian-made products.

## Quick Start

### 1. Install Dependencies

```sh
uv sync
uv run playwright install firefox
```

### 2. Set Up Environment

Create a `.env` file:
```sh
# Database
POSTGRES_HOST=localhost
POSTGRES_DB=madeinca
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# API Keys
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Click Tracking
TRACKING_ENABLED=true
TRACKING_BASE_URL=http://localhost:8000
ENVIRONMENT=development  # Set to "production" in prod
```

### 3. Run the App

**Terminal 1 - Backend API:**
```sh
uv run uvicorn backend:app --port 8000 --reload
```

**Terminal 2 - Streamlit Frontend:**
```sh
uv run streamlit run app.py
```

---

## Scraping Products

### Run All Scrapers

```sh
uv run python run_scrapes.py
```

This runs all configured brand scrapers in parallel (default: 2 concurrent). See `run_scrapes.py` for the full list of brands.

**Options:**
```sh
# Adjust parallelism
SCRAPE_MAX_PARALLEL=3 uv run python run_scrapes.py

# Add cooldown between job starts
SCRAPE_COOLDOWN_S=2.0 uv run python run_scrapes.py
```

### Scrape Individual Brands

```sh
# Shopify stores
uv run python scrape_products.py \
  --base https://provinceofcanada.com \
  --use-browser \
  --store-type shopify \
  --use-postgres

# Non-Shopify (e.g., Roots)
uv run python scrape_products.py \
  --base https://www.roots.com \
  --store-type roots \
  --use-browser \
  --url-regex='\.html' \
  --use-postgres
```

### Scrape MadeInCA Directory

```sh
uv run python scrape_madeinca.py --use-postgres --max-categories 100
```

### Supported Brands

| Brand | Status | Type |
|-------|--------|------|
| Roots | ✅ | Custom |
| Province of Canada | ✅ | Shopify |
| Manmade | ✅ | Shopify |
| Tilley | ✅ | Shopify |
| Tentree | ✅ | Shopify |
| Kamik | ✅ | Shopify |
| Sheertex | ✅ | Shopify |
| Baffin | ✅ | Shopify |
| Bushbalm | ✅ | Shopify |
| Soma Chocolate | ✅ | Shopify |
| Stanfield's | ✅ | Shopify |
| Balzac's | ✅ | Shopify |
| Muttonhead | ✅ | Shopify |
| Naked and Famous | ✅ | Shopify |
| Regimen Lab | ✅ | Shopify |
| Craig's Cookies | ✅ | Shopify |
| Jenny Bird | ✅ | Shopify |
| Green Beaver | ✅ | Shopify |
| Manitobah | ✅ | Shopify |
| Moose Knuckles | ✅ | Shopify |
| Rheo Thompson | ✅ | Shopify |
| David's Tea | ✅ | Shopify |
| Rocky Mountain Soap | ✅ | Shopify |
| Kicking Horse Coffee | ✅ | Shopify |
| St-Viateur Bagel | ✅ | Shopify |

**TODO: - brands to add**
- Canada Goose
- Lululemon
- Mejuri
- Grohmann Knives
- Aritzia
- Kotn
- Herschel 
- Ecobee
- Lacanadienne
- Joe Fresh
- purdy's chocolatier

**TODO: - re-sellers to add**
- Canadian Tire
- Sport Chek
- Mountain Equipment Co-op
- Simons
- Skiis and Bikes

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   Frontend      │     │   Backend       │     │   Database      │
│   (app.py)      │     │   (backend.py)  │     │                 │
│   :8501         │     │   :8000         │     │   :5432         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **Frontend (Streamlit)**: Chat interface, AI agent
- **Backend (FastAPI)**: Click tracking, UTM parameters, analytics
- **Database (PostgreSQL)**: Products, madeinca listings, click logs

---

## Click Tracking

All product links go through the tracking endpoint with UTM parameters:

```
/click?url=https://example.com&source=brand&product_name=Product&referrer=madeincanada.dev
```

**View stats:**
```sh
# All clicks
curl http://localhost:8000/api/clicks/stats?days=7

# Production only
curl "http://localhost:8000/api/clicks/stats?days=7&environment=production"
```

**Disable tracking:**
```sh
TRACKING_ENABLED=false
```

---

## Production Deployment

### Option A: Docker Compose (recommended)

```yaml
services:
  backend:
    build: .
    command: uvicorn backend:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: .env
    
  frontend:
    build: .
    command: streamlit run app.py --server.port 8501
    ports:
      - "8501:8501"
    env_file: .env
```

### Option B: Simple Script

```sh
#!/bin/bash
uvicorn backend:app --port 8000 &
streamlit run app.py
```
