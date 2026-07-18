#!/usr/bin/env python3
"""Retry failed scraper pages 216-235 with exponential backoff."""
import urllib.request, urllib.parse, http.cookiejar, ssl, csv, time
from pathlib import Path

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

BASE = "https://probullstats.com"
DATA_DIR = Path.home() / "dataspur" / "data"

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
    req = urllib.request.Request(
        f"{BASE}/auth/process_login.php",
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": f"{BASE}/auth/login.php",
        },
    )
    opener.open(req)
    return opener


def fetch_with_retry(opener, url, max_retries=4):
    """Exponential backoff: 1s, 2s, 4s, 8s"""
    for attempt in range(max_retries):
        try:
            resp = opener.open(url, timeout=30)
            return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            delay = min(2 ** attempt, 30)
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries} in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise


def main():
    # Load existing rides to avoid duplicates
    existing_rids = set()
    rides_path = DATA_DIR / "rides.csv"
    if rides_path.exists():
        with open(rides_path) as f:
            for row in csv.DictReader(f):
                existing_rids.add(row["rid"])

    opener = login()
    
    # Import scraper functions
    import sys
    sys.path.insert(0, str(Path.home() / "dataspur"))
    import scraper

    failed_pages = list(range(216, 236))
    new_rides = []
    new_events = []
    br = bb = sb = 0

    for page in failed_pages:
        try:
            print(f"Page {page}...", end=" ", flush=True)
            url = f"{BASE}/events/latest/{page}"
            body = fetch_with_retry(opener, url)

            events, _ = scraper.scrape_event_list(opener, page)
            
            page_added = 0
            for evt in events:
                if evt["rid"] in existing_rids:
                    continue
                try:
                    rides, stats, has_horses = scraper.scrape_event_rides(opener, evt["rid"])
                    horse = scraper.scrape_horse_rides(opener, evt["rid"]) if has_horses else []
                    all_outs = rides + horse

                    if all_outs:
                        for r in all_outs:
                            r.update({f"evt_{k}": v for k, v in evt.items()})
                            r.update({f"stat_{k}": v for k, v in stats.items()})
                            if r["event_type"] == "BR":
                                br += 1
                            elif r["event_type"] == "BB":
                                bb += 1
                            elif r["event_type"] == "SB":
                                sb += 1
                        new_rides.extend(all_outs)
                        new_events.append({**evt, "ride_count": len(all_outs), **stats})
                        page_added += len(all_outs)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"  ⚠️ {evt['rid']}: {e}", end="")

            print(f"+{len(events)} events, +{page_added} outs")

        except Exception as e:
            print(f"FAILED: {e}")

    # Append to existing CSVs
    if new_rides:
        with open(rides_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_rides[0].keys(), extrasaction="ignore")
            w.writerows(new_rides)
        print(f"\nAppended {len(new_rides)} rides (BR:{br} BB:{bb} SB:{sb}) to rides.csv")

    if new_events:
        events_path = DATA_DIR / "events.csv"
        with open(events_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_events[0].keys(), extrasaction="ignore")
            w.writerows(new_events)
        print(f"Appended {len(new_events)} events to events.csv")

    total_rides = sum(1 for _ in open(rides_path)) - 1
    total_events = sum(1 for _ in open(events_path)) - 1
    print(f"Total now: {total_events:,} events, {total_rides:,} outs")


if __name__ == "__main__":
    main()