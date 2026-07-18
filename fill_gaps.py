#!/usr/bin/env python3
"""Scrape rides for events in events.csv that have no rides in rides.csv.
Uses login from scraper.py to avoid password duplication."""
import sys, csv, time, re
from pathlib import Path

sys.path.insert(0, str(Path.home() / "dataspur"))
from scraper import login, scrape_event_rides, scrape_horse_rides

DATA_DIR = Path.home() / "dataspur" / "data"
DELAY = 0.4

def main():
    print("Ride gap filler")
    
    # Get RIDs already in rides.csv
    existing_rids = set()
    with open(DATA_DIR / "rides.csv") as f:
        for row in csv.DictReader(f):
            existing_rids.add(row["rid"])
    print(f"RIDs in rides.csv: {len(existing_rids):,}")
    
    # Get all RIDs from events.csv
    events_info = {}
    with open(DATA_DIR / "events.csv") as f:
        for row in csv.DictReader(f):
            events_info[row["rid"]] = row
    
    missing = [rid for rid in events_info if rid not in existing_rids]
    print(f"Missing events: {len(missing):,}")
    
    if not missing:
        print("No gaps!")
        return
    
    # Read existing fieldnames
    with open(DATA_DIR / "rides.csv") as f:
        fieldnames = csv.DictReader(f).fieldnames
    
    opener = login()
    total = 0
    
    for i, rid in enumerate(missing):
        evt = events_info[rid]
        
        try:
            rides, stats, has_horses = scrape_event_rides(opener, rid)
        except:
            rides = []
        
        # Enrich with event data
        for r in rides:
            r["evt_org"] = evt.get("org", "")
            r["evt_tour_class"] = evt.get("tour_class", "")
            r["evt_city"] = evt.get("city", "")
            r["evt_state"] = evt.get("state", "")
            r["evt_date"] = evt.get("date", "")
            r["evt_note"] = evt.get("note", "")
            r["evt_has_records"] = evt.get("has_records", "")
            r["evt_rid"] = rid
        
        if rides:
            with open(DATA_DIR / "rides.csv", "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writerows(rides)
            total += len(rides)
        
        if (i + 1) % 25 == 0 or (rides and len(rides) > 0):
            print(f"  {i+1}/{len(missing)}: {rid} | {len(rides)} rides | {evt.get('city','?')}, {evt.get('state','?')} | total={total:,}")
        
        time.sleep(DELAY)
    
    print(f"\nDone: {total:,} rides from {len(missing)} events")
    
    with open(DATA_DIR / "rides.csv") as f:
        final = sum(1 for _ in f) - 1
    print(f"Total rides: {final:,}")

if __name__ == "__main__":
    main()