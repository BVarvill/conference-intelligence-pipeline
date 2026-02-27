"""
Step 3: Generate Ben Leads columns and produce a ready-to-paste spreadsheet.

Takes the leads you've reviewed/selected from step2 and outputs a formatted Excel
file that can be copied directly into the Ben Leads tracking spreadsheet.

For each lead, the script:
  1. Searches Serper.dev for recent institution news and the official website URL
  2. Optionally scrapes the institution website for richer content (requires trafilatura)
  3. Calls Mistral AI to generate: Focus, Focus Summary, Area, Explanation, Video Value

Static columns calculated without AI:
  - Type (BP/MP/SP) based on budget score
  - Affinity/10 based on APA appearance count (1->5, 2->7, 3->7, 4+->10)
  - Contact/10 always 10 (never contacted)
  - Source always "Attendee List"

Output:
  {output}_ben_leads.xlsx  — two sheets: Ben Leads Ready + Research Notes
  {output}_ben_leads.csv   — CSV backup

Usage:
  python step3_prepare.py --input apa2026_extra_review.csv --output apa2026_extra

API keys (set as environment variables — see .env.example):
  MISTRAL_API_KEY, SERPER_API_KEY
"""

import argparse
import json
import os
import re
import time
import pandas as pd
import requests
from mistralai import Mistral
from datetime import date


# trafilatura enables website scraping for richer Focus accuracy.
# The script works fine without it — falls back to Serper snippets only.
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
SERPER_API_KEY  = os.environ.get("SERPER_API_KEY", "")

MISTRAL_MODEL   = "mistral-large-latest"
SERPER_ENDPOINT = "https://google.serper.dev/search"
BATCH_SIZE      = 5
DELAY           = 1.5  # seconds between Mistral batches

# Skip these when picking the official URL from search results
SOCIAL_DOMAINS = {
    "linkedin.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "youtube.com", "wikipedia.org", "indeed.com",
    "glassdoor.com", "yelp.com", "crunchbase.com", "zoominfo.com",
    "bloomberg.com", "pitchbook.com", "doximity.com", "healthgrades.com",
}

mistral_client = Mistral(api_key=MISTRAL_API_KEY)


def serper_search_inst(institution: str) -> tuple:
    """
    Search for institution news and official website in a single Serper call.
    Returns (news_snippets, website_url).
    """
    if not SERPER_API_KEY:
        return ("No news search — SERPER_API_KEY not set.", "")
    query = f'"{institution}" psychiatry mental health 2024 2025 news program'
    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        data = resp.json()

        # knowledgeGraph.website is the most reliable official URL source
        website = data.get("knowledgeGraph", {}).get("website", "")

        lines = []
        for r in data.get("organic", [])[:5]:
            lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')}")
            if not website:
                url = r.get("link", "")
                domain = url.split("/")[2].replace("www.", "") if "://" in url else ""
                if url and not any(sd in domain for sd in SOCIAL_DOMAINS):
                    website = url

        snippets = "\n".join(lines[:4]) if lines else "No recent news found."
        return (snippets, website)

    except Exception as e:
        return (f"Search error: {e}", "")


def scrape_website(url: str, max_chars: int = 1500) -> str:
    """
    Fetch and extract readable text from an institution's website (no API cost).
    Returns empty string on any failure — caller falls back to Serper snippets.
    Requires: pip install trafilatura
    """
    if not url or not HAS_TRAFILATURA:
        return ""
    try:
        html = trafilatura.fetch_url(url)
        if not html:
            return ""
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        return text[:max_chars].strip() if text else ""
    except Exception:
        return ""


def classify_type(budget_score: float) -> str:
    """Map budget score to Ben Leads type code."""
    b = float(budget_score or 0)
    if b >= 4: return "(BP) Big Private"
    if b >= 2: return "(MP) Medium Private"
    return "(SP) Small Private"


def affinity_from_appearances(appearances) -> int:
    """Map APA appearance count to Affinity/10 score."""
    apps = int(float(appearances or 0))
    if apps >= 4: return 10
    if apps >= 2: return 7
    return 5


def paste_in_crm(institution: str, focus: str, tracker_theme: str = "USA Private") -> str:
    """Format the Paste in CRM field to match the Ben Leads convention."""
    return (
        f"Call for APA\n"
        f"Tracker Theme: {tracker_theme}\n"
        f"Org: {institution}\n"
        f"Focus: {focus[:120].rstrip()}"
    )


def generate_columns(leads_ctx: list) -> list:
    """
    Generate Focus, Focus Summary, Area, Explanation, and Video Value
    for all leads via Mistral, processed in batches.

    Prefers scraped website content over Serper snippets when available.
    Returns a list of dicts in the same order as leads_ctx.
    """
    all_results = []

    for i in range(0, len(leads_ctx), BATCH_SIZE):
        batch = leads_ctx[i: i + BATCH_SIZE]
        blocks = []

        for j, ctx in enumerate(batch, 1):
            if ctx.get("scraped"):
                institution_context = (
                    f"Institution website content (use as primary source):\n{ctx['scraped']}\n\n"
                    f"Recent news snippets (secondary):\n{ctx['news']}"
                )
            else:
                institution_context = f"Recent institution news:\n{ctx['news']}"

            blocks.append(
                f"Person {j}:\n"
                f"Name: {ctx['name']}\n"
                f"Title: {ctx['title']}\n"
                f"Institution: {ctx['institution']}\n"
                f"Institution type: {ctx['inst_type']}\n"
                f"Years attended APA: {ctx['years']}\n"
                f"Number of APA appearances: {ctx['appearances']}\n"
                f"Financial capacity note: {ctx['financial_capacity']}\n"
                f"AI research notes: {ctx['reasoning']}\n"
                f"{institution_context}"
            )

        prompt = (
            "You are filling in a sales lead spreadsheet for Websedge, which produces "
            "$27,500 APA TV documentary features showcased at the American Psychiatric "
            "Association Annual Meeting (25,000+ attendees).\n\n"

            "Study these real examples from the spreadsheet to match the exact style:\n\n"

            "EXAMPLE Focus fields (3-5 sentences, lowercase start, describes what the "
            "institution does — specific programs, not generic language):\n"
            "  'focus on personalized treatment plans and telepsychiatry to increase "
            "accessibility, especially your same day access model. We also noted your pathway "
            "teams using clinical advocacy and care navigation to coordinate housing, mental "
            "health, and addiction services for vulnerable populations'\n"
            "  'measurement-based care technology platform designed to modernise psychiatric "
            "care through clinical intelligence and integrated workflows'\n\n"

            "EXAMPLE Focus Summary (2-4 words, lowercase):\n"
            "  'addiction treatment', 'telepsychiatry', 'integrated behavioral health'\n\n"

            "EXAMPLE Area:\n"
            "  'Addiction / Substance Use Disorder', 'Telepsychiatry and Digital Health', "
            "'Child and Adolescent Psychiatry', 'Inpatient Psychiatric Care', 'Clinical Research'\n\n"

            "EXAMPLE Explanation (why institution has financial capacity):\n"
            "  '84 employees', 'non-profit health system with ~$2B annual revenue', "
            "'VC-backed digital health startup, Series B funded'\n\n"

            "EXAMPLE Reason/Value/Return from video:\n"
            "  'Boosts psychiatry residency recruitment nationally and showcases their "
            "integrated care model to 25,000+ APA attendees'\n\n"

            "Generate for each person:\n"
            "1. focus — 3-5 sentences, lowercase start, specific to this institution\n"
            "2. focusSummary — 2-4 words, lowercase\n"
            "3. area — clinical area category\n"
            "4. explanation — why the institution has financial capacity\n"
            "5. videoValue — what the INSTITUTION gains from the documentary\n\n"
            f"People:\n\n{'=' * 40}\n" + f"\n{'=' * 40}\n".join(blocks) + "\n\n"
            f"Return a JSON array of exactly {len(batch)} objects. "
            "Keys: focus, focusSummary, area, explanation, videoValue\n"
            "No markdown, no code fences — raw JSON array only."
        )

        for attempt in range(6):
            try:
                resp = mistral_client.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=[
                        {"role": "system", "content":
                            "You write precise content for sales lead spreadsheets. "
                            "Match the tone and style of the examples. Return only valid JSON arrays."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                )
                text = resp.choices[0].message.content.strip()
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
                match = re.search(r"(\[.*\])", text, re.DOTALL)
                if match:
                    text = match.group(1)
                parsed = json.loads(text)
                if len(parsed) != len(batch):
                    raise ValueError(f"Expected {len(batch)}, got {len(parsed)}")
                all_results.extend(parsed)
                names = [ctx["name"] for ctx in batch]
                foci  = [r.get("focusSummary", "?") for r in parsed]
                print(f"  Batch {i // BATCH_SIZE + 1}: " +
                      " | ".join(f"{n.split()[0]}: {f}" for n, f in zip(names, foci)))
                break

            except Exception as e:
                wait = 10 * (attempt + 1)
                if attempt < 5:
                    print(f"  Attempt {attempt + 1} failed ({str(e)[:60]}) — retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  All retries failed for batch {i // BATCH_SIZE + 1}. Using placeholders.")
                    for ctx in batch:
                        all_results.append({
                            "focus": f"innovative approach to mental health care at {ctx['institution']}",
                            "focusSummary": "mental health care",
                            "area": "Mental Health",
                            "explanation": ctx.get("financial_capacity", "Financial details not available"),
                            "videoValue": "APA TV documentary provides national visibility to 25,000+ attendees.",
                        })

        time.sleep(DELAY)

    return all_results


def build_row(lead: dict, gen: dict, ctx: dict) -> dict:
    """Map enriched lead data and generated text into the Ben Leads column format."""
    today       = date.today().strftime("%Y-%m-%d")
    institution = str(lead.get("currentInstitution") or lead.get("Institution", ""))
    title       = str(lead.get("currentTitle") or lead.get("Title", ""))
    name        = str(lead.get("Name", ""))
    country     = str(lead.get("Country", ""))
    state       = str(lead.get("State", ""))
    appearances = int(float(lead.get("Appearances", 1) or 1))
    years       = str(lead.get("Years", ""))
    budget      = float(lead.get("budgetScore", 0) or 0)
    is_dm       = bool(lead.get("isDecisionMaker", False))
    confidence  = str(lead.get("confidence", "low"))
    tier        = str(lead.get("Tier", ""))
    final_score = float(lead.get("FinalScore", 0) or 0)
    flags       = str(lead.get("FLAGS", "") or "")
    website     = ctx.get("website", "")

    focus         = gen.get("focus", "")
    focus_summary = gen.get("focusSummary", "")
    area          = gen.get("area", "")
    explanation   = gen.get("explanation", "")
    video_value   = gen.get("videoValue", "")

    affinity     = affinity_from_appearances(appearances)
    affinity_exp = f"{appearances} time attendee"
    size_money   = min(int(budget * 2), 10)

    notes_parts = [f"Tier {tier}", f"Score {final_score:.1f}", f"Confidence: {confidence}"]
    if flags:
        notes_parts.append(f"Flags: {flags}")

    return {
        "Date":                           today,
        "Country":                        country,
        "State":                          state,
        "Institution":                    institution,
        "Center Name":                    "",
        "Type":                           classify_type(budget),
        "Position of Lead":               title,
        "Contact Name":                   name,
        "Email":                          "",
        "Tracker Theme":                  "USA Private",
        "Focus":                          focus,
        "Focus Summary":                  focus_summary,
        "Link":                           website,
        "Sent":                           "",
        "Area":                           area,
        "Result":                         "",
        "Pitch Rating":                   "",
        "Source":                         "Attendee List",
        "KWS Term":                       "",
        "Link to Source":                 "",
        "Attended?":                      years,
        "Affinity/10":                    affinity,
        "Affinity Explanation":           affinity_exp,
        "Size-Money/10":                  size_money,
        "Explanation":                    explanation,
        "Contacted? (Year/s)":            "",
        "Contact/10":                     10,
        "Explanation (Contact)":          "Never Contacted",
        "Potential Lead Overlap?":        "No",
        "Reason/Value/Return from video": video_value,
        "Decision Maker?":                "Yes" if is_dm else "Maybe",
        "Additional Notes":               " | ".join(notes_parts),
        "Autocheck Fellows":              0,
        "Paste in CRM":                   paste_in_crm(institution, focus),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 3: generate Ben Leads output")
    parser.add_argument("--input",  required=True,      help="Reviewed CSV/XLSX from step2")
    parser.add_argument("--output", default="prepared", help="Output filename prefix")
    args = parser.parse_args()

    print(f"\nLoading {args.input}...")
    if args.input.lower().endswith(".xlsx"):
        df = pd.read_excel(args.input, sheet_name="Review Leads")
    else:
        df = pd.read_csv(args.input)
    print(f"  {len(df)} leads to prepare")

    if not SERPER_API_KEY:
        print("  Warning: SERPER_API_KEY not set — news search disabled")
    if not HAS_TRAFILATURA:
        print("  Note: trafilatura not installed — website scraping disabled")
        print("        Run: pip install trafilatura  to enable richer Focus accuracy")

    # Build research context for each lead
    print("\nSearching institution news and scraping websites...")
    contexts = []
    for _, row in df.iterrows():
        inst = str(row.get("currentInstitution") or row.get("Institution", ""))
        news, website = serper_search_inst(inst)
        scraped = scrape_website(website) if website else ""

        fin = str(row.get("financialCapacity", "") or "")
        if not fin or fin in ("nan", "None"):
            fin = str(row.get("reasoning", "No financial data"))

        contexts.append({
            "name":               str(row.get("Name", "")),
            "title":              str(row.get("currentTitle") or row.get("Title", "")),
            "institution":        inst,
            "inst_type":          str(row.get("institutionType", "unknown")),
            "years":              str(row.get("Years", "")),
            "appearances":        row.get("Appearances", 1),
            "financial_capacity": fin,
            "reasoning":          str(row.get("reasoning", "")),
            "news":               news,
            "website":            website,
            "scraped":            scraped,
        })

        if scraped:
            source_note = f"website scraped ({len(scraped)} chars)"
        elif news and not news.startswith("No recent"):
            source_note = "Serper snippets only"
        else:
            source_note = "no source — verify manually"
        print(f"  {inst[:40]:<40}  {source_note}")
        time.sleep(0.3)

    # Generate AI text columns
    print(f"\nGenerating Ben Leads columns via Mistral ({MISTRAL_MODEL})...")
    generated = generate_columns(contexts)

    # Build output rows
    print("\nBuilding rows...")
    rows = [build_row(lead, gen, ctx)
            for lead, gen, ctx in zip(df.to_dict(orient="records"), generated, contexts)]

    df_out = pd.DataFrame(rows)

    csv_path  = f"{args.output}_ben_leads.csv"
    xlsx_path = f"{args.output}_ben_leads.xlsx"
    df_out.to_csv(csv_path, index=False)

    # Rows with no source data — highlighted orange in the output Excel
    no_news_rows = {
        i for i, ctx in enumerate(contexts)
        if not ctx.get("scraped")
        and (ctx.get("news", "") in ("No recent news found.", "")
             or ctx.get("news", "").startswith("Search error"))
    }

    # Research Notes tab shows exactly what was used to generate each Focus
    research_rows = []
    for ctx in contexts:
        news_text = ctx.get("news", "")
        scraped   = ctx.get("scraped", "")
        has_news  = news_text not in ("No recent news found.", "") and not news_text.startswith("Search error")
        if scraped:
            quality = "Website scraped — highest accuracy"
        elif has_news:
            quality = "Serper snippets only — good accuracy"
        else:
            quality = "No source data — verify Focus manually"
        research_rows.append({
            "Institution":     ctx["institution"],
            "Contact Name":    ctx["name"],
            "Website Found":   ctx.get("website", "") or "—",
            "Source Quality":  quality,
            "Website Content": (scraped[:600] + "…" if len(scraped) > 600 else scraped) or "—",
            "Serper Snippets": news_text,
        })
    df_research = pd.DataFrame(research_rows)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Ben Leads Ready", index=False)
        df_research.to_excel(writer, sheet_name="Research Notes", index=False)

        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        fill_a      = PatternFill("solid", fgColor="FFFFFF")
        fill_b      = PatternFill("solid", fgColor="F2F2F2")
        fill_nonews = PatternFill("solid", fgColor="FFE0B2")
        hdr_fill    = PatternFill("solid", fgColor="1F4E79")
        hdr_font    = Font(bold=True, color="FFFFFF", size=9)

        ws = writer.sheets["Ben Leads Ready"]
        for cell in ws[1]:
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.row_dimensions[1].height = 35

        for row_idx in range(2, ws.max_row + 1):
            data_idx = row_idx - 2
            fill = fill_nonews if data_idx in no_news_rows else (fill_a if row_idx % 2 == 0 else fill_b)
            for col_idx in range(1, ws.max_column + 1):
                ws.cell(row=row_idx, column=col_idx).fill = fill
                ws.cell(row=row_idx, column=col_idx).alignment = Alignment(wrap_text=True, vertical="top")

        col_widths = {
            "Date": 12, "Country": 8, "State": 6, "Institution": 30,
            "Center Name": 20, "Type": 16, "Position of Lead": 30,
            "Contact Name": 22, "Email": 28, "Tracker Theme": 14,
            "Focus": 60, "Focus Summary": 20, "Link": 35, "Sent": 8,
            "Area": 28, "Result": 12, "Pitch Rating": 10, "Source": 14,
            "KWS Term": 10, "Link to Source": 12, "Attended?": 20,
            "Affinity/10": 10, "Affinity Explanation": 20,
            "Size-Money/10": 10, "Explanation": 35,
            "Contacted? (Year/s)": 14, "Contact/10": 10,
            "Explanation (Contact)": 20, "Potential Lead Overlap?": 12,
            "Reason/Value/Return from video": 40,
            "Decision Maker?": 14, "Additional Notes": 35,
            "Autocheck Fellows": 12, "Paste in CRM": 40,
        }
        for col_idx, col_name in enumerate(df_out.columns, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 15)
        for row_idx in range(2, ws.max_row + 1):
            ws.row_dimensions[row_idx].height = 65
        ws.freeze_panes = "A2"

        wr = writer.sheets["Research Notes"]
        for cell in wr[1]:
            cell.fill = PatternFill("solid", fgColor="2E7D32")
            cell.font = Font(bold=True, color="FFFFFF", size=9)
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        wr.row_dimensions[1].height = 30
        for row_idx in range(2, wr.max_row + 1):
            data_idx = row_idx - 2
            for col_idx in range(1, wr.max_column + 1):
                wr.cell(row=row_idx, column=col_idx).fill = fill_nonews if data_idx in no_news_rows else fill_a
                wr.cell(row=row_idx, column=col_idx).alignment = Alignment(wrap_text=True, vertical="top")
        for col, width in [("A", 35), ("B", 22), ("C", 40), ("D", 30), ("E", 60), ("F", 70)]:
            wr.column_dimensions[col].width = width
        for row_idx in range(2, wr.max_row + 1):
            wr.row_dimensions[row_idx].height = 70
        wr.freeze_panes = "A2"

    scraped_count   = sum(1 for ctx in contexts if ctx.get("scraped"))
    no_source_count = len(no_news_rows)

    print(f"\nStep 3 complete — {len(rows)} leads prepared")
    print(f"  Website scraped: {scraped_count}")
    print(f"  No source found: {no_source_count} (orange rows — worth a manual check)")
    print(f"\nOutput: {xlsx_path}")
    print(f"  'Ben Leads Ready' tab — copy rows into Ben Leads spreadsheet")
    print(f"  'Research Notes' tab  — shows what was used to write each Focus")


if __name__ == "__main__":
    main()
