# MRA Media — Lead Enrichment Pipeline

A Python pipeline for processing APA Annual Meeting attendee lists into scored, enriched sales leads and personalised outreach emails.

Built for the WebsEdge/MRA Media sales team to identify high-value targets from conference attendee data.

## Pipeline Overview

```
Raw attendee CSV
       |
  step0_prefilter.py     → removes pharma, DNC, non-US leads instantly (no API calls)
       |
  step1_enrich.py        → live Google search + AI scoring per person (Groq + Serper)
       |
  step2_filter.py        → hard removes + flags for manual review, outputs colour-coded Excel
       |
  [manual review]        → delete rows you don't want to contact
       |
  step3_prepare.py       → generates all Ben Leads columns via Mistral, outputs ready-to-paste Excel
       |
  generate_emails.py     → writes personalised outreach emails with AI-generated hook sentences
```

## Scripts

| Script | Description |
|--------|-------------|
| `step0_prefilter.py` | Pre-filter: pharma, DNC, and non-US removal. Zero API calls. |
| `step1_enrich.py` | Enrichment: searches each person via Serper, scores them with Groq (Llama 3.3 70B). Saves checkpoints so it's safe to interrupt and resume. |
| `step2_filter.py` | Filtering: hard removes low-priority leads, flags borderline cases. Outputs colour-coded Excel. |
| `step3_prepare.py` | Preparation: generates Focus, Area, Explanation columns via Mistral. Optionally scrapes institution websites for richer context. |
| `generate_emails.py` | Email generation: personalised outreach emails with a Groq-written hook sentence per lead. |
| `merge_results.py` | Utility: merges multiple daily enriched CSVs into one ranked output. |

## Setup

```bash
pip install groq mistralai pandas tqdm requests openpyxl trafilatura
```

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Then set the environment variables before running any script:

```bash
export GROQ_API_KEY=your_key_here
export SERPER_API_KEY=your_key_here
export MISTRAL_API_KEY=your_key_here
```

## Scoring Model

FinalScore is a 0-100 composite:

| Component | Weight | Notes |
|-----------|--------|-------|
| Pre-score (attendance/authority) | 50% | From raw attendee data |
| Decision-maker score (0-5) | 35% | AI-assessed from title/role |
| Budget score (0-5) | 15% | AI-assessed from org size |

Confidence multiplier applied to AI scores: high=1.0, medium=0.7, low=0.3.

Tiers: A ≥65 | B ≥45 | C ≥25 | D <25

## Notebooks

- `APS - Correlations.ipynb` — correlation analysis of APA attendee scoring data
- `APS Machine Learning Model - Logistic Regression.ipynb` — logistic regression model for lead quality
- `APS Machine Learning Model - Random Forest.ipynb` — random forest model comparison
- `URL finder.ipynb` — scraping institution URLs from attendee data
- `WebScraper.ipynb` — institution website content extraction
- `Data Cleaning and Ranking for APA Attendees.ipynb` — initial data cleaning and pre-scoring pipeline
