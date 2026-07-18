#!/usr/bin/env python3
"""Rider Profile Scraper — extracts riding hand, career stats."""
import urllib.request, urllib.parse, http.cookiejar, re, ssl, csv, time
from pathlib import Path

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

BASE = "https://probullstats.com"
DATA_DIR = Path.home() / "dataspur" / "data"
DELAY = 0.3

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


def scrape_rider(opener, rider_slug):
    """Extract riding hand and career stats from rider profile."""
    try:
        resp = opener.open(f"{BASE}/rider/{rider_slug}", timeout=30)
        body = resp.read().decode("utf-8", errors="replace")
    except:
        return None

    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    rider = {"rider_slug": rider_slug}

    # Riding Hand
    m = re.search(r"Riding Hand:\s*(Left|Right)", text)
    rider["riding_hand"] = m.group(1) if m else None

    # Total records
    m = re.search(r"(\d+)\s+total records", text)
    rider["total_records"] = int(m.group(1)) if m else None

    # Career stats: "50% 132 rides / 264 attempts"
    m = re.search(r"Career:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["career_qual_pct"] = float(m.group(1))
        rider["career_rides"] = int(m.group(2))
        rider["career_attempts"] = int(m.group(3))

    # Career Avg Score
    m = re.search(r"Career Avg Score:\s*([\d.]+)", text)
    rider["career_avg_score"] = float(m.group(1)) if m else None

    # Career High Score
    m = re.search(r"Career High Score:\s*([\d.]+)", text)
    rider["career_high_score"] = float(m.group(1)) if m else None

    # PBR specific
    m = re.search(r"PBR:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["pbr_qual_pct"] = float(m.group(1))
        rider["pbr_rides"] = int(m.group(2))
        rider["pbr_attempts"] = int(m.group(3))

    # PBR Premier (UTB)
    m = re.search(r"PBR Premier:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["pbr_premier_qual_pct"] = float(m.group(1))
        rider["pbr_premier_rides"] = int(m.group(2))
        rider["pbr_premier_attempts"] = int(m.group(3))

    # Short Rounds
    m = re.search(r"Short Rounds:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["short_round_qual_pct"] = float(m.group(1))

    # 72+ power rating bulls
    m = re.search(r"72\+ power rating bulls:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["rank_bull_qual_pct"] = float(m.group(1))

    # Top 1000 bulls
    m = re.search(r"Top 1000 bulls:\s*([\d.]+)%\s*(\d+)\s*rides?\s*/\s*(\d+)\s*attempts", text)
    if m:
        rider["top1000_qual_pct"] = float(m.group(1))

    # Round Wins
    m = re.search(r"Round Wins vs Attempts:\s*([\d.]+)%\s*(\d+)\s*Round Wins?\s*/\s*(\d+)", text)
    if m:
        rider["round_win_pct"] = float(m.group(1))
        rider["round_wins"] = int(m.group(2))

    return rider


def main():
    print("Rider Profile Scraper")

    # Get unique rider slugs from rides data
    rider_slugs = set()
    with open(DATA_DIR / "rides.csv") as f:
        for row in csv.DictReader(f):
            slug = row.get("rider_slug", "").strip()
            if slug and row.get("event_type") == "BR":
                rider_slugs.add(slug)

    print(f"Unique riders: {len(rider_slugs):,}")

    # Skip already scraped
    out_path = DATA_DIR / "rider_profiles.csv"
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                existing.add(row["rider_slug"])
    rider_slugs -= existing
    print(f"Already scraped: {len(existing):,}, remaining: {len(rider_slugs):,}")

    if not rider_slugs:
        print("All riders scraped!")
        return

    opener = login()
    profiles = []
    fieldnames = None
    err_count = 0

    for i, slug in enumerate(sorted(rider_slugs)):
        try:
            rider = scrape_rider(opener, slug)
            if rider:
                if fieldnames is None:
                    fieldnames = list(rider.keys())
                profiles.append(rider)
        except Exception as e:
            err_count += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rider_slugs)} | {len(profiles)} scraped | {err_count} errors")
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

    print(f"\nDone: {len(existing)+i+1:,} total riders | {err_count} errors")


if __name__ == "__main__":
    main()