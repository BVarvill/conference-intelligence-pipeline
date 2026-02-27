"""
generate_emails.py
==================
Generates personalised outreach emails for APA TV leads.
Reads outreach_clear.json, uses Serper for recent company news,
Groq to write the tailored hook paragraph, outputs one email per lead.

Output: emails/ folder with one .txt file per lead + emails_all.csv
"""

import json, os, re, time, requests
from groq import Groq

# ── Keys ──────────────────────────────────────────────────────────────────────
GROQ_KEY   = os.environ.get("GROQ_API_KEY",   "")
SERPER_KEY = os.environ.get("SERPER_API_KEY", "")

groq_client = Groq(api_key=GROQ_KEY)

# ── YouTube links by category ─────────────────────────────────────────────────
# Used to pick 3-4 most relevant examples per lead
YT_LINKS = {
    "pharma": [
        "Circular Genomics: https://www.youtube.com/watch?v=M0ie-UvxMxM",
        "CVS Health: https://www.youtube.com/watch?v=bDjIn5KBvGA",
        "Medibio: https://www.youtube.com/watch?v=aPNX-nWH-hQ",
        "Koa Health: https://www.youtube.com/watch?v=LfVL0-S1lhk",
        "Ginger: https://www.youtube.com/watch?v=DUqee6mPKe0",
    ],
    "digital_health": [
        "Talkiatry: https://www.youtube.com/watch?v=eMWYa-JmMqw",
        "Koa Health: https://www.youtube.com/watch?v=LfVL0-S1lhk",
        "Ginger: https://www.youtube.com/watch?v=DUqee6mPKe0",
        "Conscious Health: https://www.youtube.com/watch?v=UADZqaN5T7I",
    ],
    "hospital": [
        "Novant Health Psychiatry Institute: https://www.youtube.com/watch?v=RwqQu-KapTQ",
        "WellSpan Philhaven Behavioral Health Network: https://www.youtube.com/watch?v=nveCTx7t6OY",
        "MetroHealth Behavioral Health Center: https://www.youtube.com/watch?v=LV8D3opombg",
        "Penn State Health, Department of Psychiatry and Behavioral Health: https://www.youtube.com/watch?v=Gz3eJCa77eM",
        "Sheppard Pratt: https://www.youtube.com/watch?v=u-FKtHh-YbI",
        "Maimonides Health, Department of Psychiatry: https://www.youtube.com/watch?v=jYdTgr-rs1Q",
    ],
    "research": [
        "OHSU Center for Mental Health Innovation: https://www.youtube.com/watch?v=5L2oF2xG4RA",
        "Indiana University School of Medicine, IU Addiction Psychiatry Program: https://www.youtube.com/watch?v=i2O5Qz7revM",
        "Geisinger, Department of Psychiatry and Behavioral Health: https://www.youtube.com/watch?v=_1XtcyzPAM8",
        "University of California, San Diego, The Psychedelics and Health Research Initiative: https://www.youtube.com/watch?v=hbky-FOGUJE",
    ],
}

def pick_links(institution_type: str, title: str, institution: str) -> str:
    """Pick 3-4 most relevant YouTube examples based on institution type."""
    t = institution_type.lower()
    ti = (title + " " + institution).lower()

    if t == "company":
        # Pharma vs digital health
        if any(w in ti for w in ["pharma","therapeut","bioscience","biotech","drug","medicine","clinical","biogen","otsuka","janssen","alkermes","takeda","sage","neurocrine","indivior"]):
            links = YT_LINKS["pharma"][:4]
        else:
            links = YT_LINKS["digital_health"][:4]
    else:
        # Hospital - mix hospital + research
        links = YT_LINKS["hospital"][:3] + YT_LINKS["research"][:1]

    return "\n".join(links)


def serper_search(institution: str, name: str) -> str:
    """Fetch recent news snippets about the institution."""
    query = f"{institution} psychiatry mental health news 2025 2026 innovation"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        data = resp.json()
        lines = []
        for r in data.get("organic", [])[:5]:
            lines.append(f"- {r.get('title','')}: {r.get('snippet','')}")
        return "\n".join(lines) if lines else "No recent news found."
    except Exception as e:
        return f"Search error: {e}"


def generate_hook(name: str, last_name: str, title: str, institution: str,
                  inst_type: str, reasoning: str, snippets: str) -> str:
    """Use Groq to write the tailored hook paragraph (the 'We were curious about...' sentence)."""

    prompt = f"""You are writing ONE sentence for a sales email from Ben Varvill at WebsEdge.
WebsEdge produces documentary videos ($27,500) showcased at the APA Annual Meeting.
The email is to invite {name} ({title} at {institution}) to be profiled in an APA TV documentary.

Context about their work (from web search):
{snippets}

Additional notes: {reasoning}

Write ONLY the hook sentence that replaces "We were curious about your latest XXXXXX" in this template:
"We were curious about your latest XXXXXX, which we believe will make an exceptional fit with our focus areas at the APA Annual Meeting this year."

Rules:
- Replace XXXXXX with something SPECIFIC and REAL about {institution}'s work (a programme, product, initiative, or recent achievement)
- Use the search results above to make it genuinely specific — not generic
- Keep it to ONE sentence, completing the template exactly
- Do NOT include "We were curious about your latest" — just provide the replacement text for XXXXXX
- Be specific: e.g. "innovative same-day access psychiatric care model" not "mental health work"
- Output ONLY the replacement phrase, nothing else"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You write precise, specific one-phrase hooks for sales emails. Output only the requested phrase."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        hook = response.choices[0].message.content.strip().strip('"').strip("'")
        # Clean up any accidental full sentence starts
        hook = re.sub(r'^(We were curious about your latest\s*)', '', hook, flags=re.IGNORECASE).strip()
        return hook
    except Exception as e:
        return f"[their work in {institution}]"


def get_salutation(name: str, title: str) -> str:
    """Work out Dear Dr. X vs Dear First Name."""
    # If they have MD, PhD, Director, Chief etc - use Dr.
    medical_titles = ["md", "m.d.", "phd", "ph.d.", "psychiatrist", "physician",
                      "medical director", "chief medical", "clinical"]
    if any(t in title.lower() for t in medical_titles):
        last = name.strip().split()[-1]
        return f"Dr. {last}"
    first = name.strip().split()[0]
    return first


def build_email(lead: dict, hook: str) -> str:
    """Assemble the full email from Dear to Ben."""
    name       = lead["fullName"]
    title      = lead["title"]
    institution = lead["institution"]
    inst_type  = lead["institutionType"]
    salutation = get_salutation(name, title)
    links      = pick_links(inst_type, title, institution)

    email = f"""Dear {salutation},

I would like to schedule a call between you and Mark Rose, APA TV director, to discuss potentially highlighting {institution} in a pre-recorded video case study within the official broadcast at the American Psychiatric Association (APA) 2026 Annual Meeting in San Francisco (May 16-20, 2026) and online.

We were curious about your latest {hook}, which we believe will make an exceptional fit with our focus areas at the APA Annual Meeting this year.

As I'm sure you are aware, the APA Annual Meeting offers the largest audience of psychiatrists and mental health professionals at any meeting in the world. The APA has again partnered with WebsEdge, following 14 extremely successful years, to produce APA TV, the official broadcast of the APA Annual Meeting. This enables an important platform to showcase to the attendees some of the latest ground-breaking innovations across research, training, technology and patient care that are helping to shape the future of psychiatry and mental health.

As a key part of the APA TV broadcast, we will once again be highlighting to the attendees some companies, psychiatry departments and health systems that are at the cutting-edge of mental health innovation and offer them a unique opportunity to profile their key research, recruitment initiatives and best practices in the form of a five-minute documentary feature.

Through our research, we are considering a number of companies, hospitals and health systems as potential candidates to sponsor these documentary features including {institution}, and I am keen to arrange a conversation between you and Mark Rose to make sure that there is a strong fit. I must emphasise that there is a cost involved in this opportunity to be profiled, which covers the production, distribution, and full ownership of the film and all additional footage.

As part of this project, any partner profiled in this way will retain the rights to the finished film and all footage we shoot. Mark will of course run through this and all logistics with you in more detail during the call.

In advance of the conversation, it would be useful for you to have a look at one or two of the groups that we profiled at recent APA Annual Meetings as this will give you an idea on the style of film we would produce with you. You can see a few of those films here:

{links}

As such, please could you email back with some suitable times over the next few days when Mark can call you to discuss this? He will be in meetings for the majority of today, but is fairly open over the next few days if you can suggest a couple of times for an initial call?

I look forward to hearing back from you with a convenient time to speak.

Best wishes,

Ben"""
    return email


def main():
    base = "/Users/benvarvill/Downloads/MRA Media work"
    out_dir = os.path.join(base, "emails")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base, "outreach_clear.json")) as f:
        leads = json.load(f)

    # Filter out private practice (too small, not companies/hospitals in the target sense)
    leads = [l for l in leads if "private practice" not in l["institution"].lower()]
    print(f"\n📧 Generating emails for {len(leads)} leads...\n")

    results = []
    for i, lead in enumerate(leads, 1):
        name        = lead["fullName"]
        institution = lead["institution"]
        title       = lead["title"]

        print(f"[{i}/{len(leads)}] {name} @ {institution}")

        # 1. Search for recent news
        snippets = serper_search(institution, name)

        # 2. Generate hook
        last_name = name.strip().split()[-1]
        hook = generate_hook(
            name, last_name, title, institution,
            lead["institutionType"], lead.get("reasoning",""), snippets
        )
        print(f"         Hook: {hook[:80]}...")

        # 3. Build email
        email_text = build_email(lead, hook)

        # 4. Save individual file
        safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
        filepath = os.path.join(out_dir, f"{i:03d}_{safe_name}.txt")
        with open(filepath, "w") as f:
            f.write(email_text)

        results.append({
            "rank": i,
            "name": name,
            "title": title,
            "institution": institution,
            "institutionType": lead["institutionType"],
            "tier": lead.get("tier",""),
            "finalScore": lead.get("finalScore",""),
            "hook": hook,
            "email": email_text,
            "crmNote": lead.get("crmNote",""),
        })

        time.sleep(1.5)  # Serper rate limit

    # Save CSV of all emails
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(base, "emails_draft.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Done! {len(results)} emails generated")
    print(f"   Individual files: {out_dir}/")
    print(f"   All-in-one CSV:   {csv_path}")


if __name__ == "__main__":
    main()
