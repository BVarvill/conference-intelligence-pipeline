"""
Step 1: Enrich and score attendee leads using live search + AI analysis.

Takes the pre-filtered attendee CSV from step0 and enriches each record with:
  - Serper.dev — live Google search results per person
  - Groq (Llama 3.3 70B) — scores budget authority and decision-maker level

Scoring weights:
  Pre-score (attendance/authority):  50%
  Budget score (org financial size): 15%  — reduced because AI often struggles with small org revenue
  Decision-maker score:              35%  — increased because role/title is reliably assessable

Checkpoint saves after every batch so it's safe to Ctrl-C and resume.

Usage:
  python step1_enrich.py --input apa2026_extra_step1_input.csv --output apa2026_extra

API keys (set as environment variables — see .env.example):
  GROQ_API_KEY, SERPER_API_KEY, MISTRAL_API_KEY
"""

import argparse
import json
import os
import re
import time
import pandas as pd
import requests
from tqdm import tqdm
from groq import Groq
from mistralai import Mistral


# API keys — set via environment variables, never hardcode
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
SERPER_API_KEY  = os.environ.get("SERPER_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# Model config — falls back from primary Groq -> secondary Groq -> Mistral on quota exhaustion
GROQ_MODEL           = "llama-3.3-70b-versatile"
FALLBACK_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
MISTRAL_BACKUP_MODEL = "mistral-large-latest"

BATCH_SIZE            = 5
DELAY_BETWEEN_BATCHES = 2
MAX_RETRIES           = 8
SERPER_ENDPOINT       = "https://google.serper.dev/search"

# Scoring weights — must sum to 1.0
WEIGHT_PRE_SCORE = 0.50
WEIGHT_BUDGET    = 0.15
WEIGHT_DECISION  = 0.35

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.3}

# Input column names
COL_NAME        = "Name"
COL_INSTITUTION = "Institution"
COL_TITLE       = "Title"
COL_COUNTRY     = "Country"
COL_PRE_SCORE   = "FinalPreScore"

# API clients (initialised once at module level)
groq_client    = Groq(api_key=GROQ_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Tracks which model we're currently routing to
_active_model = GROQ_MODEL
_use_mistral  = False


def _extract_json(text: str) -> list:
    """Parse a JSON array from the model response, handling markdown code fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[.*\])", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from response: {text[:300]}")


def _error_record(name: str) -> dict:
    """Placeholder record used when enrichment fails for a person."""
    return {
        "originalName": name,
        "currentTitle": "Error",
        "currentInstitution": "Error",
        "institutionType": "unknown",
        "financialCapacity": "N/A",
        "financialRaw": 0,
        "budgetScore": 0,
        "decisionMakerScore": 0,
        "isDecisionMaker": False,
        "reasoning": "Processing failed.",
        "confidence": "low",
    }


def clean_institution(name: str) -> str:
    """Strip leading numeric IDs sometimes present in the raw data."""
    return re.sub(r"^\d+-", "", str(name or "")).strip()


def serper_search(name: str, institution: str, country: str) -> str:
    """Run a Google search via Serper and return formatted result snippets."""
    if not SERPER_API_KEY:
        return "No search results — SERPER_API_KEY not set."
    query = f"{name} {institution} psychiatry current role 2024 2025"
    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 4},
            timeout=10,
        )
        results = resp.json().get("organic", [])[:4]
        lines = [f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results]
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def enrich_batch(records: list) -> list:
    """
    Send a batch of people to the LLM for enrichment and scoring.
    Handles quota errors by stepping down through the model fallback chain:
    GROQ_MODEL -> FALLBACK_MODEL -> Mistral.
    """
    global _active_model, _use_mistral

    people_blocks = []
    for r in records:
        name     = r.get(COL_NAME, "")
        inst     = clean_institution(r.get(COL_INSTITUTION, ""))
        title    = r.get(COL_TITLE, "")
        country  = r.get(COL_COUNTRY, "")
        snippets = serper_search(name, inst, country)
        people_blocks.append(
            f"Name: {name}\nLast known title: {title}\n"
            f"Last known institution: {inst}\nCountry: {country}\n"
            f"Live Google search results:\n{snippets}"
        )

    prompt = f"""You are a professional research analyst for Websedge, a conference media company.
These are past attendees of the American Psychiatric Association (APA) annual conference.
Use the live Google search results to determine their CURRENT role and employer.

Scoring rules:
- budgetScore 0-5 (organisation financial capacity):
    0=unknown, 1=<$1M, 2=$1M-$10M, 3=$10M-$100M, 4=$100M-$1B, 5=>$1B
- decisionMakerScore 0-5 (authority to approve a $27,500 spend):
    0=unknown, 1=student/resident/junior, 2=mid-level, 3=senior staff/researcher,
    4=director/VP/department head, 5=CEO/president/CMO/C-suite
- isDecisionMaker: true if decisionMakerScore >= 4
- confidence: "high" / "medium" / "low"
- institutionType: company / university / research_center / government / nonprofit / hospital / unknown
- financialCapacity: plain-English org size e.g. "Non-profit health system, ~$2B revenue"
- financialRaw: estimated USD numeric value (0 if unknown)
- reasoning: 1-2 sentences explaining your scores

People to analyse:

{"---".join(people_blocks)}

Return ONLY a valid JSON array of exactly {len(records)} objects in the SAME ORDER as the input.
Keys: originalName, currentTitle, currentInstitution, institutionType,
      financialCapacity, financialRaw, budgetScore, decisionMakerScore,
      isDecisionMaker, reasoning, confidence
No markdown, no prose — raw JSON array only."""

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            if _use_mistral:
                resp = mistral_client.chat.complete(
                    model=MISTRAL_BACKUP_MODEL,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON arrays — no markdown, no prose."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                text = resp.choices[0].message.content
            else:
                response = groq_client.chat.completions.create(
                    model=_active_model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON arrays — no markdown, no prose."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                text = response.choices[0].message.content

            if not text:
                raise ValueError("Empty response")

            results = _extract_json(text)

            # Pad with error records if the model returned fewer than expected
            while len(results) < len(records):
                results.append(_error_record(records[len(results)].get(COL_NAME, "?")))

            return results

        except Exception as e:
            err = str(e)
            attempt += 1

            # Step down the fallback chain on quota/rate limit errors
            quota_phrases = [
                "tokens per day", "daily token", "rate limit", "resource_exhausted",
                "quota", "429", "too many requests", "exceeded", "capacity",
            ]
            if any(p in err.lower() for p in quota_phrases) and not _use_mistral:
                if _active_model != FALLBACK_MODEL:
                    _active_model = FALLBACK_MODEL
                    print(f"\n  Groq daily quota hit — switching to {FALLBACK_MODEL}")
                    time.sleep(2)
                    continue
                else:
                    _use_mistral = True
                    print(f"\n  Both Groq quotas exhausted — switching to Mistral ({MISTRAL_BACKUP_MODEL})")
                    time.sleep(2)
                    continue

            wait_match = re.search(r"try again in (\d+)m(\d+)", err)
            wait = int(wait_match.group(1)) * 60 + float(wait_match.group(2)) + 5 if wait_match else 15 * attempt

            if attempt < MAX_RETRIES:
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed — waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print("  All retries exhausted — using error placeholders.")
                return [_error_record(r.get(COL_NAME, "?")) for r in records]


def compute_final_score(row: pd.Series) -> float:
    """
    Combines pre-score, budget score, and decision-maker score into a 0-100 final score.

    FinalScore = (
        (FinalPreScore / 100) * 0.50
      + (budgetScore / 5) * confidence_weight * 0.15
      + (dmScore / 5) * confidence_weight * 0.35
    ) * 100
    """
    pre      = float(row.get(COL_PRE_SCORE, 0) or 0)
    budget   = float(row.get("budgetScore", 0) or 0)
    decision = float(row.get("decisionMakerScore", 0) or 0)
    conf_w   = CONFIDENCE_WEIGHTS.get(str(row.get("confidence", "low")).lower(), 0.3)

    return round((
        (pre / 100)    * WEIGHT_PRE_SCORE +
        (budget / 5)   * conf_w * WEIGHT_BUDGET +
        (decision / 5) * conf_w * WEIGHT_DECISION
    ) * 100, 2)


def assign_tier(score: float) -> str:
    if score >= 65: return "A"
    if score >= 45: return "B"
    if score >= 25: return "C"
    return "D"


def main():
    parser = argparse.ArgumentParser(description="Step 1: enrich and score leads")
    parser.add_argument("--input",  required=True,          help="Input attendee CSV (from step0)")
    parser.add_argument("--output", default="enriched",     help="Output filename prefix")
    parser.add_argument("--limit",  type=int, default=None, help="Only process first N records")
    parser.add_argument("--offset", type=int, default=0,    help="Skip first N records")
    args = parser.parse_args()

    checkpoint_path = f"{args.output}_checkpoint.json"

    print(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input)

    if args.offset:
        df = df.iloc[args.offset:].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    records = df.to_dict(orient="records")
    total   = len(records)
    batches = [records[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    # Resume from checkpoint if one exists
    all_enrichments = []
    start_batch = 0
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found — resuming...")
        with open(checkpoint_path) as f:
            ck = json.load(f)
        all_enrichments = ck.get("enrichments", [])
        start_batch = len(all_enrichments) // BATCH_SIZE
        print(f"  {len(all_enrichments)} records already done, resuming from batch {start_batch + 1}")
    else:
        print(f"  {total} records | {len(batches)} batches | Model: {_active_model}")

    print()
    for batch_num, batch in enumerate(
        tqdm(batches[start_batch:], desc="Enriching", unit="batch"),
        start=start_batch + 1
    ):
        enrichments = enrich_batch(batch)
        all_enrichments.extend(enrichments)

        for r, e in zip(batch, enrichments):
            inst = str(e.get("currentInstitution") or "?")
            print(f"  {r.get(COL_NAME, '?'):<35} -> {inst:<40} "
                  f"B:{e.get('budgetScore', 0)} DM:{e.get('decisionMakerScore', 0)} [{e.get('confidence', '?')}]")

        with open(checkpoint_path, "w") as f:
            json.dump({"enrichments": all_enrichments}, f)

        if batch_num < len(batches):
            time.sleep(DELAY_BETWEEN_BATCHES)

    # Merge enrichment results back into the original dataframe
    print("\nMerging and scoring...")
    enr_df = pd.DataFrame(all_enrichments)
    for col in enr_df.columns:
        df[col] = enr_df[col].values if col in enr_df.columns else None

    df["FinalScore"] = df.apply(compute_final_score, axis=1)
    df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    df["Tier"] = df["FinalScore"].apply(assign_tier)
    df.insert(0, "Rank", df.index + 1)

    csv_path  = f"{args.output}_enriched.csv"
    json_path = f"{args.output}_enriched.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    tiers = df["Tier"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    print(f"\nEnrichment complete — {total} records processed")
    print(f"  Tier A (>=65): {tiers['A']}")
    print(f"  Tier B (>=45): {tiers['B']}")
    print(f"  Tier C (>=25): {tiers['C']}")
    print(f"  Tier D (<25):  {tiers['D']}")
    print(f"\nOutput: {csv_path}")
    print(f"Next:   python step2_filter.py --input {csv_path}")


if __name__ == "__main__":
    main()
