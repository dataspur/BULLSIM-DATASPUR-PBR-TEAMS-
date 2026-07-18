#!/usr/bin/env python3
"""Re-scrape horse events (BB/SB) with fixed rider slug and horse ID extraction."""
import csv, time, sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "dataspur"))
from scraper import login, scrape_horse_rides

DATA = Path.home() / "dataspur" / "data"
DELAY = 0.4

def main():
    # Get all event RIDs that have horse rides
    event_rids = set()
    with open(DATA / "rides.csv") as f:
        for row in csv.DictReader(f):
            if row["event_type"] in ("BB", "SB"):
                event_rids.add(row["rid"])
    
    print(f"Horse events to re-scrape: {len(event_rids)}")
    
    # Read existing fieldnames
    with open(DATA / "rides.csv") as f:
        fieldnames = csv.DictReader(f).fieldnames
    
    opener = login()
    
    # Find event data for enrichment
    events_map = {}
    with open(DATA / "events.csv") as f:
        for row in csv.DictReader(f):
            events_map[row["rid"]] = row
    
    total = 0
    new_rides = []
    
    for i, rid in enumerate(sorted(event_rids)):
        evt = events_map.get(rid, {})
        
        try:
            rides = scrape_horse_rides(opener, rid)
        except Exception as e:
            print(f"  {rid}: ERROR {e}")
            rides = []
        
        for r in rides:
            r["evt_org"] = evt.get("org", "")
            r["evt_tour_class"] = evt.get("tour_class", "")
            r["evt_city"] = evt.get("city", "")
            r["evt_state"] = evt.get("state", "")
            r["evt_date"] = evt.get("date", "")
            r["evt_note"] = evt.get("note", "")
            r["evt_has_records"] = evt.get("has_records", "")
            r["evt_rid"] = rid
        
        new_rides.extend(rides)
        total += len(rides)
        
        if (i + 1) % 20 == 0:
            # Check extraction quality
            slugs_ok = sum(1 for r in new_rides if r["rider_slug"])
            horses_ok = sum(1 for r in new_rides if r["bull_id"])
            print(f"  {i+1}/{len(event_rids)} | {total} rides | "
                  f"slugs={slugs_ok}/{total} horses={horses_ok}/{total}")
        
        time.sleep(DELAY)
    
    print(f"\nRe-scraped {total} horse rides from {len(event_rids)} events")
    slugs_ok = sum(1 for r in new_rides if r["rider_slug"])
    print(f"  Rider slugs: {slugs_ok}/{total} ({slugs_ok/max(total,1)*100:.1f}%)")
    horses_ok = sum(1 for r in new_rides if r["bull_id"])
    print(f"  Horse IDs: {horses_ok}/{total} ({horses_ok/max(total,1)*100:.1f}%)")
    
    # Now: remove old horse rides and append new ones
    print("\nReplacing horse rides in CSV...")
    non_horse = []
    with open(DATA / "rides.csv") as f:
        for row in csv.DictReader(f):
            if row["event_type"] not in ("BB", "SB"):
                non_horse.append(row)
    
    # Write combined
    import os, tempfile, shutil
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=DATA)
    os.close(fd)
    
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(non_horse)
        w.writerows(new_rides)
    
    os.replace(tmp, DATA / "rides.csv")
    
    with open(DATA / "rides.csv") as f:
        final = sum(1 for _ in f) - 1
    print(f"Done. Rides: {final:,} (BR: {len(non_horse):,} + horse: {total:,})")

if __name__ == "__main__":
    main()