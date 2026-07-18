#!/usr/bin/env python3
"""
Bull Profile Scraper — extracts power ratings, spin direction, handedness splits.
~7,900 bulls × 0.3s = ~40 minutes.
"""
import urllib.request, urllib.parse, http.cookiejar, re, ssl, csv, time
from pathlib import Path
from collections import defaultdict

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

BASE = "https://probullstats.com"
DATA_DIR = Path.home() / "dataspur" / "data"
DELAY = 0.25

USERNAME = "mhx5pkbv@anonaddy.com"
PASSWORD = "Carlye3$"


def login():
    cj = http.cookiejar.MozillaCookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ssl_ctx),
    )
    opener.open(f"{BASE}/auth/login.php")
    data = urllib.parse.urlencode(
        {"username": USERNAME, "password": PASSWORD, "formname": "loginform"}
    ).encode()
    opener.open(urllib.request.Request(
        f"{BASE}/auth/process_login.php", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Referer": f"{BASE}/auth/login.php"},
    ))
    return opener


def scrape_bull(opener, bull_id):
    """Extract all stats from a bull profile page."""
    try:
        resp = opener.open(f"{BASE}/bulls/{bull_id}", timeout=30)
        body = resp.read().decode("utf-8", errors="replace")
    except:
        return None

    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    bull = {"bull_id": bull_id}

    # Name from title
    name = re.search(r"<title>([^<]+)</title>", body)
    if name:
        bull["name"] = name.group(1).replace(" Bull Profile", "").strip()

    # PBS Power Rating
    m = re.search(r"PBS Power Rating:\s*([\d.]+)", text)
    bull["power_rating"] = float(m.group(1)) if m else None

    # Avg bull score (per judge: 0-25 range)
    m = re.search(r"Avg:\s*([\d.]+)", text)
    bull["avg_bull_score"] = float(m.group(1)) if m else None

    # Adjusted avg
    m = re.search(r"Adjusted:\s*([\d.]+)", text)
    bull["avg_adjusted"] = float(m.group(1)) if m else None

    # Attempts
    m = re.search(r"(\d+)\s+Attempts", text)
    bull["attempts"] = int(m.group(1)) if m else None

    # Rides (qualified rides against)
    m = re.search(r"(\d+)\s+rides\b", text)
    bull["rides"] = int(m.group(1)) if m else None

    # Avg Ride Score
    m = re.search(r"Avg Ride Score:\s*([\d.]+)", text)
    bull["avg_ride_score"] = float(m.group(1)) if m else None

    # Buckoff percentage
    m = re.search(r"([\d.]+)% buckoff percentage", text)
    bull["buckoff_pct"] = float(m.group(1)) if m else None

    # Pre-Ride Probability
    m = re.search(r"Pre-Ride Probability:\s*([\d.]+)%", text)
    bull["pre_ride_prob"] = float(m.group(1)) if m else None

    # Riding Hand Advantage
    m = re.search(r"Riding Hand Advantage:\s*(\d+)%\s*(LEFT|RIGHT)", text)
    if m:
        bull["hand_advantage_pct"] = int(m.group(1))
        bull["hand_advantage_dir"] = m.group(2)

    # Left-hander record: "Left handers 7-8 46.67%"
    m = re.search(r"Left handers?\s+(\d+)-(\d+)\s+([\d.]+)%", text)
    if m:
        bull["lh_wins"] = int(m.group(1))  # rider wins (qualified rides)
        bull["lh_losses"] = int(m.group(2))  # bull wins (buckoffs)
        bull["lh_pct"] = float(m.group(3))
        bull["lh_total"] = bull["lh_wins"] + bull["lh_losses"]

    # Right-hander record: "Right handers 11-7 61.11%"
    m = re.search(r"Right handers?\s+(\d+)-(\d+)\s+([\d.]+)%", text)
    if m:
        bull["rh_wins"] = int(m.group(1))
        bull["rh_losses"] = int(m.group(2))
        bull["rh_pct"] = float(m.group(3))
        bull["rh_total"] = bull["rh_wins"] + bull["rh_losses"]

    # Top riders record
    m = re.search(r"Top Riders\s+(\d+)-(\d+)\s+([\d.]+)%", text)
    if m:
        bull["top_riders_wins"] = int(m.group(1))
        bull["top_riders_losses"] = int(m.group(2))
        bull["top_riders_pct"] = float(m.group(3))

    # Power Rating Breakdown
    m = re.search(r"Difficulty[:\s]*([\d.]+)", text)
    if m:
        bull["pr_difficulty"] = float(m.group(1))
    m = re.search(r"Bull Score Potential[:\s]*([\d.]+)", text)
    if m:
        bull["pr_score_potential"] = float(m.group(1))
    m = re.search(r"Experience[:\s]*([\d.]+)", text)
    if m:
        bull["pr_experience"] = float(m.group(1))

    # Contractor
    m = re.search(r"profile\s+\S+\s+(.*?)\s+PBS:", text)
    if m:
        bull["contractor"] = m.group(1).strip()

    # Active status
    bull["active"] = "active:" in text.lower()
    
    # PBR UTB stats
    m = re.search(r"PBR UTB:\s*(\d+)\s+outs?:\s*(\d+)-(\d+)\s*\(([\d.]+)\s+avg\)", text)
    if m:
        bull["pbr_utb_outs"] = int(m.group(1))
        bull["pbr_utb_record"] = f"{m.group(2)}-{m.group(3)}"
        bull["pbr_utb_avg"] = float(m.group(4))

    # Round Wins
    m = re.search(r"Round Wins:\s*(\d+)", text)
    if m:
        bull["round_wins"] = int(m.group(1))
    m = re.search(r"Rider Round Wins:\s*(\d+)", text)
    if m:
        bull["rider_round_wins"] = int(m.group(1))

    return bull


def main():
    print("Bull Profile Scraper")
    
    # Get all unique bull IDs from rides data
    bull_ids = set()
    with open(DATA_DIR / "rides.csv") as f:
        for row in csv.DictReader(f):
            bid = row.get("bull_id", "").strip()
            if bid and row.get("event_type") == "BR":
                # Handle float-formatted IDs like "47633.0"
                try:
                    bid_int = str(int(float(bid)))
                    bull_ids.add(bid_int)
                except (ValueError, TypeError):
                    if bid.isdigit():
                        bull_ids.add(bid)

    print(f"Unique bulls to scrape: {len(bull_ids):,}")
    
    # Skip already scraped
    out_path = DATA_DIR / "bull_profiles.csv"
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                existing.add(row["bull_id"])
    bull_ids -= existing
    print(f"Already scraped: {len(existing):,}, remaining: {len(bull_ids):,}")
    
    if not bull_ids:
        print("All bulls already scraped!")
        return

    opener = login()
    profiles = []
    fieldnames = None
    err_count = 0

    for i, bid in enumerate(sorted(bull_ids)):
        try:
            bull = scrape_bull(opener, bid)
            if bull:
                if fieldnames is None:
                    fieldnames = list(bull.keys())
                profiles.append(bull)
        except Exception as e:
            err_count += 1
            if err_count <= 5:
                print(f"  Error {bid}: {e}")

        # Progress
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(bull_ids)} bulls | {len(profiles)} scraped | {err_count} errors")
            # Save progress
            if profiles and fieldnames:
                mode = "a" if out_path.exists() else "w"
                with open(out_path, mode, newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    if mode == "w":
                        w.writeheader()
                    w.writerows(profiles)
                profiles = []

        time.sleep(DELAY)

    # Final save
    if profiles:
        mode = "a" if out_path.exists() else "w"
        with open(out_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                w.writeheader()
            w.writerows(profiles)

    print(f"\nDone: {len(existing)+i+1:,} total bulls | {err_count} errors")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()