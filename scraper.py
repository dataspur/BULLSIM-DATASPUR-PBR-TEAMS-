#!/usr/bin/env python3
"""
DataSpur — Probullstats Scraper v3
Scrapes all events: bull riding (BR), bareback (BB), saddle bronc (SB).
"""
import urllib.request, urllib.parse, http.cookiejar, re, ssl, csv, time
from pathlib import Path
from datetime import datetime

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

BASE = "https://probullstats.com"
DATA_DIR = Path.home() / "dataspur" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    req = urllib.request.Request(
        f"{BASE}/auth/process_login.php",
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": f"{BASE}/auth/login.php",
        },
    )
    resp = opener.open(req)
    if "fail" in resp.geturl():
        raise RuntimeError(f"Login failed: {resp.geturl()}")
    print("Authenticated")
    return opener


def scrape_event_list(opener, page):
    url = f"{BASE}/events/latest/{page}" if page > 1 else f"{BASE}/events/"
    resp = opener.open(url)
    body = resp.read().decode("utf-8", errors="replace")

    events = []
    # Pattern: <a href='/events/event.php?rid=XXX'><li class='... ORG'>... city, ST - date ...</li></a>
    for m in re.finditer(
        r"<a\s+href=['\"]/events/event\.php\?rid=([^'\"]+)['\"][^>]*>"
        r"\s*<li\s+class=['\"][^'\"]*(PRCA|PBR|TEAMS|TDP|CPRA|CBR|ABBI|OTHER)[^'\"]*['\"][^>]*>"
        r"(.*?)"
        r"</li>\s*</a>",
        body,
        re.DOTALL,
    ):
        rid = m.group(1)
        org_raw = m.group(2)
        content = m.group(3)

        # Clean HTML from content
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Parse: "PRCA:  Colorado Springs, CO - Jul, 2026 (NFR Open) **"
        # Remove leading org label: "PRCA:"
        text = re.sub(r"^\w+\s*:\s*", "", text).strip()

        # Extract note from parens at end
        note = ""
        note_match = re.search(r"\(([^)]+)\)\s*$", text)
        if note_match:
            note = note_match.group(1)
            text = text[: note_match.start()].strip()

        # Check for ** records marker
        has_records = "**" in text
        text = text.replace("**", "").strip()

        # Parse: "Colorado Springs, CO - Jul, 2026"
        parts = [p.strip() for p in text.split(",")]
        city = parts[0] if len(parts) > 0 else ""
        state_date = parts[1] if len(parts) > 1 else ""
        year_str = parts[2] if len(parts) > 2 else ""
        sd_parts = [p.strip() for p in state_date.split("-")]
        state = sd_parts[0] if len(sd_parts) > 0 else ""
        date_str = sd_parts[1].strip() if len(sd_parts) > 1 else ""
        if year_str:
            date_str = f"{date_str}, {year_str}"

        # Map org
        if org_raw == "TEAMS":
            org = "PBR"
            tour_class = "Team Series"
        elif org_raw == "TDP":
            org = "PBR"
            tour_class = "Touring Pro"
        else:
            org = org_raw
            tour_class = org_raw

        events.append(
            {
                "rid": rid,
                "org": org,
                "tour_class": tour_class,
                "city": city,
                "state": state,
                "date": date_str,
                "note": note,
                "has_records": has_records,
            }
        )

    pages_match = re.search(r"Page \d+ of (\d+)", body)
    total = int(pages_match.group(1)) if pages_match else 1
    return events, total


def scrape_event_rides(opener, rid):
    """Scrape bull riding outs (&view=outs)."""
    url = f"{BASE}/events/event.php?rid={rid}&view=outs"
    try:
        resp = opener.open(url, timeout=30)
    except Exception:
        return [], {}, False

    body = resp.read().decode("utf-8", errors="replace")

    # Stats
    stats_text = re.sub(r"<[^>]+>", " ", body)
    stats = {}
    for key, pattern in [
        ("records", r"Records\s+(\d+)"),
        ("official_outs", r"Official Outs\s+(\d+)"),
        ("avg_bull_score", r"Avg Bull Score\s+([\d\.]+)"),
        ("qualified_rides", r"Qualified Rides\s+(\d+)\s+\((\d+)%\)"),
    ]:
        m = re.search(pattern, stats_text)
        if m:
            if key == "qualified_rides":
                stats["qualified_rides"] = int(m.group(1))
                stats["qualified_pct"] = int(m.group(2))
            elif key == "avg_bull_score":
                stats[key] = float(m.group(1))
            else:
                stats[key] = int(m.group(1))

    # Detect horse records (skip "0 horse records")
    horse_match = re.search(r"(\d+)\s+horse records", stats_text)
    has_horses = bool(horse_match) and int(horse_match.group(1)) > 0
    horse_count = int(horse_match.group(1)) if horse_match else 0

    # Extract ride rows
    rides = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.DOTALL):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
        if len(tds) < 7:
            continue

        perf_raw = re.sub(r"<[^>]+>", "", tds[0]).strip()
        if not perf_raw.isdigit():
            continue

        go_raw = re.sub(r"<[^>]+>", "", tds[1]).strip()

        match_html = tds[2]
        rider_match = re.search(r'/rider/([^"\']+)[^>]*>([^<]+)<', match_html)
        bull_match = re.search(r"/bulls/(\d+)[^>]*>([^<]+)<", match_html)

        rider_slug = rider_match.group(1) if rider_match else ""
        rider_name = rider_match.group(2).strip() if rider_match else ""
        bull_id = bull_match.group(1) if bull_match else ""
        bull_name = bull_match.group(2).strip() if bull_match else ""

        score_raw = re.sub(r"<[^>]+>", "", tds[3]).strip()
        score = (
            float(score_raw)
            if score_raw
            and score_raw.replace(".", "").replace("-", "").isdigit()
            and score_raw != "-"
            else None
        )

        bull_html = tds[4]
        bull_score_raw = re.split(r'<br[^>]*>', bull_html)[0]
        bull_score_raw = re.sub(r'<[^>]+>', '', bull_score_raw).strip()
        bull_score = (
            float(bull_score_raw)
            if bull_score_raw
            and bull_score_raw.replace(".", "").replace("-", "").isdigit()
            and bull_score_raw != "-"
            else None
        )

        judge_scores = re.search(r"<br[^>]*>\s*([\d\.\|]+)", bull_html)
        judge_scores = judge_scores.group(1).strip() if judge_scores else ""

        time_raw = re.sub(r"<[^>]+>", "", tds[5]).strip()
        ride_time = (
            float(time_raw)
            if time_raw and time_raw.replace(".", "").isdigit()
            else None
        )

        comments = re.sub(r"<[^>]+>", "", tds[6]).strip()
        qualified = score is not None and score > 0

        rides.append(
            {
                "rid": rid,
                "perf": int(perf_raw),
                "go": int(go_raw) if go_raw.isdigit() else 1,
                "rider_slug": rider_slug,
                "rider_name": rider_name,
                "bull_id": bull_id,
                "bull_name": bull_name,
                "score": score,
                "qualified": qualified,
                "bull_score": bull_score,
                "judge_scores": judge_scores,
                "ride_time": ride_time,
                "comments": comments,
                "event_type": "BR",
                "stock_score": bull_score,
                "ride_plus_minus": None,
            }
        )

    return rides, stats, has_horses


def scrape_horse_rides(opener, rid):
    """Scrape bareback (BB) and saddle bronc (SB) horse outs."""
    url = f"{BASE}/events/event.php?rid={rid}&view=horse-outs"
    try:
        resp = opener.open(url, timeout=30)
    except Exception:
        return []

    body = resp.read().decode("utf-8", errors="replace")
    horse_rides = []

    # Detect which section we're in (BB or SB) by looking for headers
    # The page has sections like "Bareback Riding" and "Saddle Bronc Riding"
    # Each section has rows: round perf horse_name contr rider score stock_score ride_pm
    current_event = None

    # Split into BB and SB sections
    sections = re.split(
        r"(Bareback Riding|Saddle Bronc Riding)", body
    )
    for i in range(1, len(sections), 2):
        section_label = sections[i].strip()
        section_body = sections[i + 1]

        if "Bareback" in section_label:
            current_event = "BB"
        elif "Saddle" in section_label:
            current_event = "SB"
        else:
            continue

        # Find all rows in this section
        for tr in re.findall(
            r"<tr[^>]*>(.*?)</tr>", section_body, re.DOTALL
        ):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
            if len(tds) < 7:
                continue

            # Parse columns: round, perf, horse_name, contr, rider, score, stock_score, ride_pm
            # Extract rider slug and horse ID from links before stripping HTML
            rider_slug = ""
            rider_match = re.search(r'/riders/broncrider\.php\?cid=(\d+)', tds[4])
            if rider_match:
                rider_slug = "cid_" + rider_match.group(1)
            
            horse_id = ""
            horse_match = re.search(r'/bulls/horse\.php\?pbid=(\d+)', tds[2])
            if horse_match:
                horse_id = "pbid_" + horse_match.group(1)
            
            round_raw = re.sub(r"<[^>]+>", "", tds[0]).strip()
            perf_raw = re.sub(r"<[^>]+>", "", tds[1]).strip()
            horse_raw = re.sub(r"<[^>]+>", "", tds[2]).strip()
            contr_raw = re.sub(r"<[^>]+>", "", tds[3]).strip()
            rider_raw = re.sub(r"<[^>]+>", "", tds[4]).strip()
            score_raw = re.sub(r"<[^>]+>", "", tds[5]).strip()
            stock_raw = re.sub(r"<[^>]+>", "", tds[6]).strip()

            # Ride +/- may be in col 7 or 8 depending on format
            rpm_raw = ""
            if len(tds) >= 8:
                rpm_raw = re.sub(r"<[^>]+>", "", tds[7]).strip()

            if not round_raw.isdigit():
                continue

            # Parse values
            perf_val = int(perf_raw) if perf_raw.isdigit() else None
            round_val = int(round_raw) if round_raw.isdigit() else 1

            score_val = (
                float(score_raw)
                if score_raw
                and score_raw.replace(".", "").replace("-", "").isdigit()
                and score_raw not in ("", "-")
                else None
            )

            stock_score_val = (
                float(stock_raw)
                if stock_raw
                and stock_raw.replace(".", "").replace("-", "").isdigit()
                and stock_raw not in ("", "-")
                else None
            )

            rpm_val = (
                float(rpm_raw)
                if rpm_raw
                and rpm_raw.replace(".", "").replace("-", "").isdigit()
                and rpm_raw not in ("", "-")
                else None
            )

            qualified = score_val is not None and score_val > 0

            horse_rides.append(
                {
                    "rid": rid,
                    "perf": perf_val,
                    "go": round_val,
                    "rider_slug": rider_slug,
                    "rider_name": rider_raw,
                    "bull_id": horse_id,
                    "bull_name": horse_raw,
                    "score": score_val,
                    "qualified": qualified,
                    "bull_score": None,
                    "judge_scores": "",
                    "ride_time": None,
                    "comments": contr_raw,  # contractor shortcode
                    "event_type": current_event,
                    "stock_score": stock_score_val,
                    "ride_plus_minus": rpm_val,
                }
            )

    return horse_rides


# ── MAIN ────────────────────────────────────────────
def main():
    print(f"DataSpur Scraper v3 (BR + BB + SB) — {datetime.now()}")
    opener = login()

    _, total_pages = scrape_event_list(opener, 1)
    print(f"Total pages: {total_pages} (~{total_pages * 50:,} events)")

    all_rides = []
    all_events = []
    br_count = bb_count = sb_count = 0

    for page in range(1, min(total_pages + 1, 236)):
        try:
            events, _ = scrape_event_list(opener, page)
        except Exception as e:
            print(f"  Page {page}: {e}")
            continue

        for evt in events:
            try:
                # Scrape bull riding outs
                rides, stats, has_horses = scrape_event_rides(opener, evt["rid"])

                # Scrape horse outs if present
                horse_rides = []
                if has_horses:
                    horse_rides = scrape_horse_rides(opener, evt["rid"])

                all_outs = rides + horse_rides

                if all_outs:
                    for r in all_outs:
                        r.update({f"evt_{k}": v for k, v in evt.items()})
                        r.update({f"stat_{k}": v for k, v in stats.items()})

                    # Count by type
                    for r in all_outs:
                        if r["event_type"] == "BR":
                            br_count += 1
                        elif r["event_type"] == "BB":
                            bb_count += 1
                        elif r["event_type"] == "SB":
                            sb_count += 1

                    all_rides.extend(all_outs)
                    all_events.append(
                        {**evt, "ride_count": len(all_outs), **stats}
                    )

                time.sleep(DELAY)

            except Exception as e:
                print(f"  Event {evt['rid']}: {e}")

        if page % 25 == 0:
            print(
                f"  Page {page}/{total_pages} | "
                f"{len(all_events):,} events | {len(all_rides):,} outs "
                f"(BR:{br_count:,} BB:{bb_count:,} SB:{sb_count:,})"
            )

    # Save
    print(f"\n{'='*50}")
    print(
        f"COMPLETE: {len(all_events):,} events, {len(all_rides):,} outs "
        f"(BR:{br_count:,} BB:{bb_count:,} SB:{sb_count:,})"
    )

    if all_rides:
        FIELD_ORDER = [
            "rid", "event_type", "perf", "go",
            "rider_slug", "rider_name", "bull_id", "bull_name",
            "score", "qualified", "bull_score", "stock_score",
            "ride_plus_minus", "judge_scores", "ride_time", "comments",
            "evt_org", "evt_tour_class", "evt_city", "evt_state",
            "evt_date", "evt_note", "evt_has_records",
            "stat_records", "stat_official_outs", "stat_avg_bull_score",
            "stat_qualified_rides", "stat_qualified_pct",
        ]
        # Include any extra fields not in the ordered list
        all_keys = set()
        for r in all_rides:
            all_keys.update(r.keys())
        fieldnames = [f for f in FIELD_ORDER if f in all_keys]
        fieldnames += sorted(all_keys - set(FIELD_ORDER))

        rides_file = DATA_DIR / "rides.csv"
        with open(rides_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rides)
        print(f"  rides.csv ({len(all_rides):,} outs)")

    if all_events:
        events_file = DATA_DIR / "events.csv"
        with open(events_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_events[0].keys())
            w.writeheader()
            w.writerows(all_events)
        print(f"  events.csv ({len(all_events):,} events)")

    return all_rides, all_events


if __name__ == "__main__":
    main()