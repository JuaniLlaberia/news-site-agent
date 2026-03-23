# News Sites Analyzer Agent

## Overview

**News Sites Analyzer Agent** is an intelligent multi-agent system built with Flask and LangGraph that automates the analysis and scraping configuration extraction of news websites. Given a URL, the system uses a pipeline of AI agents to inspect the site's DOM structure, identify CSS selectors for key content fields, measure safe request rates, and validate the resulting configuration by running a live test scrape through an external scraper service.

The end result is a ready-to-use scraping configuration and an optional PDF report documenting the entire analysis — no manual DOM inspection required.

---

## What It Does

1. **Inspects the target website** — A Web Inspector agent (powered by Google Gemini) loads and analyzes the site's HTML structure to extract CSS selectors for titles, content, subtitles, authors, images, and dates, as well as a map of relevant URL paths.
2. **Measures rate limits** — A Rate Limit Analyzer probes the site to determine a safe request delay, avoiding bans or blocks during scraping.
3. **Validates the configuration** — The extracted config is sent to an external scraper service, which performs a real test scrape and confirms whether the selectors work correctly.
4. **Generates a PDF report** — A Reporter agent (powered by Ollama by default) produces a detailed PDF documenting the analysis process, observations, and final configuration.

---

## What It Can Be Used For

- **Bootstrapping news scrapers** — Automatically produce scraping configs for new outlets, eliminating the need to manually inspect HTML.
- **Scraper pipeline onboarding** — Feed the output config directly into a downstream scraper that ingests news articles at scale.
- **Media monitoring tooling** — Quickly add new sources to a media monitoring or press aggregation system.
- **Research pipelines** — Automate discovery and configuration of sources for NLP datasets, media bias analysis, or event detection systems.

> **Note:** The system is purpose-built around news outlet structure (articles, authors, publication dates, etc.), but with moderate modifications to the selector extraction logic it can be adapted to analyze and configure scrapers for other content-heavy websites.

---

## Architecture

The system is orchestrated via LangGraph and composed of three main services:

```
┌──────────────────────────────────────┐
│         News Sites Agent             │
│  ┌────────────────────────────────┐  │
│  │         Orchestrator           │  │
│  │  web_inspector → rate_limit_   │  │
│  │  tester → tester               │  │
│  └────────────┬───────────────────┘  │
└───────────────┼──────────────────────┘
                │
    ┌───────────▼──────────┐   ┌─────────────┐
    │   Gemini (Analysis)  │   │   Ollama    │
    │  Web Inspector Agent │   │  (Reports)  │
    └──────────────────────┘   └─────────────┘
                │
    ┌───────────▼──────────┐
    │  External Scraper    │
    │  (Validation Step)   │
    └──────────────────────┘
```

### Agent Pipeline

| Agent | Role |
|---|---|
| **Web Inspector** | Analyzes HTML/DOM; extracts CSS selectors and URL map using Gemini |
| **Rate Limit Analyzer** | Probes site to determine safe request delay |
| **Tester** | Calls the external scraper service to validate the config |
| **Reporter** | Generates a PDF report via sub-agents (Planner, Writer, Concluder) |

---

## Dependencies & External Services

### Google Gemini (Required)
The Web Inspector agent uses Gemini for DOM analysis and selector extraction. A valid Google API key is required — the system will not function without it.

### Ollama (Required for PDF reports — swappable)
The Reporter agent uses a local Ollama instance (default model: `gemma2:2b`) to generate narrative PDF reports. Ollama can be replaced with any other LLM provider by modifying the reporter agent's model configuration. If you don't need PDF reports, this dependency is optional.

### External Scraper Service (Required)
The validation step depends on an **external scraper service** running at `SCRAPER_URL` (default: `http://scraper:5000`). This service receives the extracted configuration and performs a live test scrape to confirm the selectors are correct. **You must provide and run this scraper service separately** — it is not included in this repository. The agent will not be able to validate configurations without it.

---

## API Endpoints

### Health Check

**GET** `/`

```json
{ "status": "healthy" }
```

---

### Analyze News Site

**POST** `/`

Analyzes a news website and returns its scraping configuration.

**Request**
```json
{
  "name": "Example News",
  "url": "https://example-news.com"
}
```

**Response (200)**
```json
{
  "config": {
    "name": "Example News",
    "base_url": "https://example-news.com",
    "url_dict": { "home": "/", "articles": "/articles" },
    "title_selector": "h1.article-title",
    "content_selector": ["div.article-content", "p.paragraph"],
    "subtitle_selector": "h2.article-subtitle",
    "author_selector": "span.author-name",
    "img_url_selector": "img.featured-image",
    "date_selector": "time.publish-date",
    "rate_limit": 1.5
  },
  "observations": [...]
}
```

**Error Responses**
- `400` — Missing `name` or `url`, or `url` not using HTTPS
- `500` — Server error during analysis

---

### Generate Report

**POST** `/generate_report`

Generates a PDF report from a previously returned config and observation list.

**Request**
```json
{
  "config": { ... },
  "observations": [ ... ]
}
```

**Response (200)** — PDF binary stream (`Content-Type: application/pdf`)

**Error Responses**
- `400` — Missing `config` or `observations`
- `500` — Config validation failed

---

## Installation

### Prerequisites

- Python 3.11+
- Playwright (for browser automation)
- Ollama running locally (for report generation)
- An external scraper service deployed and accessible
- A valid Google API key for Gemini

### Environment Variables

Create a `.env` file in the root:

```bash
# Required — Google Gemini
GEMINI_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_gemini_api_key

# Optional — Gemini model override
GEMINI_MODEL=gemini-2.0-flash-exp

# Optional — Ollama (report generation)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b

# Optional — Backup Gemini keys
GOOGLE_API_KEY_BACKUP_1=your_backup_key_1
GOOGLE_API_KEY_BACKUP_2=your_backup_key_2

# Optional — Flask
FLASK_ENV=production
PYTHONUNBUFFERED=1

# Required — External scraper service
SCRAPER_URL=http://scraper:5000
```

### Running with Docker Compose

The system is designed to run via Docker Compose alongside Ollama and your external scraper service:

```bash
docker compose up
```

The API will be available at `http://localhost:5000`.  
Swagger docs are available at `http://localhost:5000/swagger`.

---

## Swapping the Report LLM

The Reporter agent uses Ollama by default, but it can be replaced with any LLM provider. To change it, update the model configuration in the reporter agent and adjust the `OLLAMA_BASE_URL` / `OLLAMA_MODEL` env vars (or replace the client entirely). Gemini or any OpenAI-compatible API can be used as a drop-in alternative.
