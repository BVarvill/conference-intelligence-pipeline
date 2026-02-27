"""
enrich_leads.py
===============
Websedge Conference Lead Enrichment Pipeline
---------------------------------------------
Takes a pre-scored attendee CSV and enriches each record using Groq (Llama 3.3 70B)
with Serper.dev Google Search results to find:
  - Current institution / company
  - Current job title / role
  - Organisation financial capacity / valuation
  - Decision-maker likelihood

Outputs:
  - enriched_leads.csv   (sorted by FinalScore descending)
  - enriched_leads.json  (same data, full fidelity)

Usage:
  python enrich_leads.py --input your_attendees.csv --output enriched_all

Requirements:
  pip install groq pandas tqdm requests

Environment variables:
  GROQ_API_KEY    — Groq API key (14,400 free req/day on free tier)
  SERPER_API_KEY  — Serper.dev API key (Google search results, 2,500 free/month)

FinalScore formula:
  FinalScore = (
      (FinalPreScore / 100) × 0.50   [attendance/authority from your data]
    + (budgetScore / 5) × conf_w × 0.25   [org financial capacity, 0-5]
    + (decisionMakerScore / 5) × conf_w × 0.25   [authority to approve spend, 0-5]
  ) × 100

  Confidence weights: high=1.0, medium=0.7, low=0.3

Tiers:
  A = score >= 65  (email first)
  B = score >= 45  (worth contacting)
  C = score >= 25  (possible)
  D = score <  25  (low priority)
"""

import argparse
import json
import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL            = "llama-3.3-70b-versatile"  # 14,400 free req/day
BATCH_SIZE            = 5       # records per API call
DELAY_BETWEEN_BATCHES = 2       # seconds — Groq is fast, small pause to be safe
MAX_RETRIES           = 3
RETRY_DELAY           = 5       # seconds between retries on failure
SERPER_ENDPOINT       = "https://google.serper.dev/search"

# Weights for FinalScore (must sum to 1.0)
WEIGHT_PRE_SCORE = 0.50   # your existing FinalPreScore
WEIGHT_BUDGET    = 0.25   # AI budget/valuation score
WEIGHT_DECISION  = 0.25   # AI decision-maker score

# Confidence multipliers applied to AI scores
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.3}

# ── Column names (exactly as they appear in your CSV) ─────────────────────────

COL_NAME           = "Name"
COL_INSTITUTION    = "Institution"
COL_TITLE          = "Title"
COL_COUNTRY        = "Country"
COL_STATE          = "State"
COL_APPEARANCES    = "Appearances"
COL_YEARS          = "Years"
COL_FREQ_SCORE     = "FrequencyScore"
COL_RECENCY_SCORE  = "RecencyScore"
COL_MOMENTUM_SCORE = "MomentumScore"
COL_AUTHORITY_SCORE= "AuthorityScore"
COL_PRE_SCORE      = "FinalPreScore"

# ── API client setup ───────────────────────────────────────────────────────────

groq_key   = os.environ.get("GROQ_API_KEY")
serper_key = os.environ.get("SERPER_API_KEY")

if not groq_key:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable not set.\n"
        "Run: export GROQ_API_KEY='your-key-here'"
    )

groq_client = Groq(api_key=groq_key)

# ── Helper: extract JSON from LLM response ────────────────────────────────────

def _extract_json(text: str) -> list:
    """Extract a JSON array from a response that may contain markdown or prose."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find the first [...] block
    match = re.search(r"(\[.*\])", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON array from response: {text[:300]}")


# ── Serper web search ─────────────────────────────────────────────────────────

def serper_search(name: str, last_institution: str, country: str) -> str:
    """
    Fetch top 4 Google search results for a person via Serper.dev.
    Returns a plain-text snippet block to embed in the Groq prompt.
    Falls back to a note string if Serper is unavailable or key is missing.
    """
    if not serper_key:
        return "No search results (SERPER_API_KEY not set)."
    query = f"{name} {last_institution} psychiatry current role"
    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": 4},
            timeout=10,
        )
        data = resp.json()
        lines = []
        for r in data.get("organic", [])[:4]:
            lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')}")
        return "\n".join(lines) if lines else "No search results found."
    except Exception as e:
        return f"Search error: {e}"


# ── Error placeholder ─────────────────────────────────────────────────────────

def _error_placeholder(name: str) -> dict:
    """Returns a safe fallback enrichment record for a person that couldn't be processed."""
    return {
        "originalName":       name,
        "currentTitle":       "Error — could not process",
        "currentInstitution": "Error — could not process",
        "institutionType":    "unknown",
        "financialCapacity":  "N/A",
        "financialRaw":       0,
        "budgetScore":        0,
        "decisionMakerScore": 0,
        "isDecisionMaker":    False,
        "reasoning":          "Processing failed for this record.",
        "confidence":         "low",
    }


# ── Institution name cleaner ──────────────────────────────────────────────────

def clean_institution(name: str) -> str:
    """Strip numeric codes like '33-' or '11-' prepended to institution names."""
    return re.sub(r"^\d+-", "", str(name or "")).strip()


# ── Core enrichment function ──────────────────────────────────────────────────

def enrich_batch(records: list[dict]) -> list[dict]:
    """
    Enrich a batch using Groq (Llama 3.3 70B) with Serper-provided search results.
    Returns a list of enrichment dicts in the same order as input.
    """
    # Build per-person search context with live snippets
    people_blocks = []
    for r in records:
        name     = r.get(COL_NAME, "")
        inst     = clean_institution(r.get(COL_INSTITUTION, ""))
        title    = r.get(COL_TITLE, "")
        country  = r.get(COL_COUNTRY, "")
        snippets = serper_search(name, inst, country)
        people_blocks.append(
            f"Name: {name}\n"
            f"Last known title: {title}\n"
            f"Last known institution: {inst}\n"
            f"Country: {country}\n"
            f"Live Google search results:\n{snippets}"
        )

    people_text = "\n\n---\n\n".join(people_blocks)

    prompt = f"""You are a professional research analyst for Websedge, a conference media company.
These people are past attendees of the American Psychiatric Association (APA) annual conference.
They are psychiatrists, mental health researchers, hospital administrators, and healthcare executives.
Using the live Google search results provided for each person, determine their CURRENT role and employer.

Scoring rules:
- budgetScore 0-5:
    0 = unknown
    1 = <$1M budget (very unlikely to afford $27,500 video)
    2 = $1M-$10M (possible stretch)
    3 = $10M-$100M (comfortable)
    4 = $100M-$1B (easily affordable)
    5 = >$1B (trivially affordable)
- decisionMakerScore 0-5:
    0 = unknown
    1 = student / junior staff (no authority)
    2 = mid-level (limited influence)
    3 = senior staff / senior researcher (some authority)
    4 = director / head of centre / VP (likely decision-maker)
    5 = CEO / president / C-suite / PI of major centre (clear decision-maker)
- isDecisionMaker: true if decisionMakerScore is 4 or 5
- confidence: "high", "medium", or "low" — how certain are you given the search results?

People to analyse:

{people_text}

Return ONLY a valid JSON array with exactly {len(records)} objects in the SAME ORDER as the input.
Each object must have these exact keys:
  originalName, currentTitle, currentInstitution, institutionType,
  financialCapacity, financialRaw, budgetScore, decisionMakerScore,
  isDecisionMaker, reasoning, confidence

institutionType must be one of: company, university, research_center, government, nonprofit, hospital, unknown
financialCapacity: plain-English org size e.g. "R1 university, $800M endowment" or ">$1B revenue"
financialRaw: estimated USD numeric value of the org (0 if unknown)
reasoning: 1-2 sentences explaining your budget and decision-maker assessment

No markdown, no prose, no code fences — raw JSON array only."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise research analyst. "
                            "Always base assessments on the provided search results. "
                            "Return only valid JSON arrays — no markdown, no prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            text = response.choices[0].message.content
            if not text:
                raise ValueError("Empty response from Groq")

            results = _extract_json(text)

            if len(results) != len(records):
                print(f"  ⚠ Expected {len(records)} results, got {len(results)}. Padding.")
                while len(results) < len(records):
                    results.append(_error_placeholder(records[len(results)].get(COL_NAME, "Unknown")))

            return results

        except Exception as e:
            print(f"  ✗ Groq attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("  ✗ All retries exhausted for this batch. Using error placeholders.")
                return [_error_placeholder(r.get(COL_NAME, "Unknown")) for r in records]


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_final_score(row: pd.Series) -> float:
    """
    Combines the existing FinalPreScore with AI budget and decision scores.
    AI scores are discounted by confidence level so uncertain results rank lower.

    Formula:
      FinalScore = (
          (FinalPreScore/100) × 0.50
        + (budgetScore/5) × confidence_weight × 0.25
        + (decisionMakerScore/5) × confidence_weight × 0.25
      ) × 100

    Confidence weights: high=1.0, medium=0.7, low=0.3
    """
    pre      = float(row.get(COL_PRE_SCORE, 0) or 0)
    budget   = float(row.get("budgetScore", 0) or 0)
    decision = float(row.get("decisionMakerScore", 0) or 0)
    conf_key = str(row.get("confidence", "low") or "low").lower()
    conf_w   = CONFIDENCE_WEIGHTS.get(conf_key, 0.3)

    pre_norm      = pre / 100.0
    budget_norm   = (budget / 5.0) * conf_w
    decision_norm = (decision / 5.0) * conf_w

    final = (
        pre_norm    * WEIGHT_PRE_SCORE +
        budget_norm * WEIGHT_BUDGET    +
        decision_norm * WEIGHT_DECISION
    ) * 100

    return round(final, 2)


def assign_tier(score: float) -> str:
    """A/B/C/D tier label for immediate actionability."""
    if score >= 65: return "A"
    if score >= 45: return "B"
    if score >= 25: return "C"
    return "D"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Websedge Lead Enrichment Pipeline (Groq + Serper)")
    parser.add_argument("--input",  required=True,  help="Path to input CSV file")
    parser.add_argument("--output", default="enriched_leads", help="Output filename prefix (no extension)")
    parser.add_argument("--limit",  type=int, default=None, help="Only process first N rows (for testing)")
    parser.add_argument("--offset", type=int, default=0,    help="Skip the first N rows")
    args = parser.parse_args()

    checkpoint_path = f"{args.output}_checkpoint.json"

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\n📂 Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"   {len(df)} total records in file.")

    if args.offset:
        df = df.iloc[args.offset:].reset_index(drop=True)
        print(f"   ⚙ Offset — skipping first {args.offset} records.")

    if args.limit:
        df = df.head(args.limit)
        print(f"   ⚙ Limit — processing {args.limit} records.")

    records = df.to_dict(orient="records")
    total   = len(records)
    batches = [records[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    # ── Load checkpoint if one exists ─────────────────────────────────────────
    all_enrichments = []
    start_batch     = 0

    if os.path.exists(checkpoint_path):
        print(f"\n♻️  Checkpoint found: {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_enrichments = checkpoint.get("enrichments", [])
        start_batch     = len(all_enrichments) // BATCH_SIZE
        already_done    = len(all_enrichments)
        print(f"   Resuming from batch {start_batch + 1} ({already_done} records already done)")
    else:
        print(f"\n🔍 Starting enrichment — {total} records in {len(batches)} batches of {BATCH_SIZE}")
        print(f"   Model: {GROQ_MODEL} | Search: Serper.dev")

    print()

    # ── Process batches ───────────────────────────────────────────────────────
    for batch_num, batch in enumerate(
        tqdm(batches[start_batch:], desc="Enriching", unit="batch"),
        start=start_batch + 1
    ):
        enrichments = enrich_batch(batch)
        all_enrichments.extend(enrichments)

        # Per-record progress line
        names = [r.get(COL_NAME, "?") for r in batch]
        for name, result in zip(names, enrichments):
            conf   = result.get("confidence", "?")
            budget = result.get("budgetScore", 0)
            dm     = result.get("decisionMakerScore", 0)
            inst   = result.get("currentInstitution", "?")
            print(f"   ✓ {name:<35} → {inst:<40} B:{budget} DM:{dm} [{conf}]")

        # Save checkpoint after every batch (safe to Ctrl-C and resume)
        with open(checkpoint_path, "w") as f:
            json.dump({"enrichments": all_enrichments}, f)

        if batch_num < len(batches):
            time.sleep(DELAY_BETWEEN_BATCHES)

    # ── Merge enrichments back into main dataframe ────────────────────────────
    print("\n📊 Merging enrichment data...")

    enrichment_cols = [
        "currentTitle", "currentInstitution", "institutionType",
        "financialCapacity", "financialRaw",
        "budgetScore", "decisionMakerScore", "isDecisionMaker",
        "reasoning", "confidence"
    ]

    enrichment_df = pd.DataFrame(all_enrichments)
    for col in enrichment_cols:
        df[col] = enrichment_df[col].values if col in enrichment_df.columns else None

    # ── Compute scores and tiers ───────────────────────────────────────────────
    df["FinalScore"] = df.apply(compute_final_score, axis=1)
    df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    df["Tier"] = df["FinalScore"].apply(assign_tier)
    df.insert(0, "Rank", df.index + 1)

    # ── Write output files ─────────────────────────────────────────────────────
    csv_path  = f"{args.output}.csv"
    json_path = f"{args.output}.json"

    print(f"\n💾 Writing outputs...")
    df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV  → {csv_path}")

    df.to_json(json_path, orient="records", indent=2)
    print(f"   ✓ JSON → {json_path}")

    # Remove checkpoint now that we have the final files
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"   ✓ Checkpoint removed")

    # ── Summary stats ──────────────────────────────────────────────────────────
    decision_makers = df[df["isDecisionMaker"] == True]
    high_budget     = df[df["budgetScore"] >= 4]
    top_leads       = df[df["FinalScore"] >= 70]
    tiers           = df["Tier"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)

    print(f"""
╔══════════════════════════════════════════╗
║           ENRICHMENT COMPLETE            ║
╠══════════════════════════════════════════╣
║  Total records processed : {total:<14} ║
║  Decision makers found   : {len(decision_makers):<14} ║
║  High budget orgs (4–5)  : {len(high_budget):<14} ║
║  Top leads (score ≥ 70)  : {len(top_leads):<14} ║
╠══════════════════════════════════════════╣
║  Tier A (email first)    : {tiers['A']:<14} ║
║  Tier B (worth contacting: {tiers['B']:<14} ║
║  Tier C (possible)       : {tiers['C']:<14} ║
║  Tier D (low priority)   : {tiers['D']:<14} ║
╚══════════════════════════════════════════╝
    """)

    print("🏆 Top 10 Leads:")
    top10_cols = ["Rank", "Tier", COL_NAME, "currentTitle", "currentInstitution",
                  "budgetScore", "decisionMakerScore", "FinalScore"]
    available = [c for c in top10_cols if c in df.columns]
    print(df[available].head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
