"""
fill_ncaab_results.py
Fills final scores into ncaab-log-YYYY-MM-DD.json (written by
log_ncaab_predictions.js) and grades each logged lean/take against the line
that was logged with it: outcome = 'win' | 'loss' | 'push'.

Games are matched by ESPN event id, so there is no team-name matching here.

Set BACKFILL_DAYS=N to also process the N-1 days before DATE, so a gap in
results heals itself on the next run.
"""

import json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

import gist_api

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')
backfill = max(1, int(os.environ.get('BACKFILL_DAYS', '1')))

GIST_USER = 'loganthein'
ESPN_URL  = ('https://site.api.espn.com/apis/site/v2/sports/basketball/'
             'mens-college-basketball/scoreboard?groups=50&limit=300&dates={}')

def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/{GIST_USER}/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'Note: could not fetch {filename}: {e}')
        return None

def fetch_finals(date_str):
    """ESPN event id → (homeScore, awayScore) for finished games. None if the fetch failed."""
    url = ESPN_URL.format(date_str.replace('-', ''))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'fill-ncaab-results'})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f'ESPN fetch failed: {e}', file=sys.stderr)
        return None
    finals = {}
    for ev in d.get('events', []):
        comp = (ev.get('competitions') or [{}])[0]
        if comp.get('status', {}).get('type', {}).get('name') != 'STATUS_FINAL':
            continue
        cs   = comp.get('competitors', [])
        home = next((c for c in cs if c.get('homeAway') == 'home'), None)
        away = next((c for c in cs if c.get('homeAway') == 'away'), None)
        if home and away:
            finals[ev['id']] = (int(home.get('score') or 0), int(away.get('score') or 0))
    return finals

def grade(pick, home_score, away_score):
    if pick['market'] == 'spread':
        margin = home_score - away_score
        if not pick['sideIsHome']:
            margin = -margin
        diff = margin + pick['line']
    else:
        total = home_score + away_score
        diff = (total - pick['line']) if pick['side'] == 'over' else (pick['line'] - total)
    return 'win' if diff > 0 else 'loss' if diff < 0 else 'push'

def fill_date(date_str):
    """Returns True on a clean pass."""
    filename = f'ncaab-log-{date_str}.json'
    log = gist_fetch(filename)
    if not log:
        print(f'{date_str}: no log file — nothing to fill.')
        return True
    if all(e.get('result') is not None for e in log):
        print(f'{date_str}: all games already have results.')
        return True

    finals = fetch_finals(date_str)
    if finals is None:
        return False

    updated = 0
    for e in log:
        if e.get('result') is not None or e['game_id'] not in finals:
            continue
        hs, as_ = finals[e['game_id']]
        for p in e['picks']:
            p['outcome'] = grade(p, hs, as_)
        e['result'] = {'homeScore': hs, 'awayScore': as_}
        updated += 1
        graded = ', '.join(f"{p['tier']} {p['market']} {p['outcome']}" for p in e['picks']) or 'no picks'
        print(f"Filled: {e['game']} → {as_}-{hs} ({graded})")

    if not updated:
        print(f'{date_str}: no newly finished games.')
        return True
    try:
        print(f'Patching {filename} ({updated} results)… '
              f'status {gist_api.patch(gist_id, gist_pat, filename, json.dumps(log, indent=2), "fill-ncaab-results")}')
    except Exception as e:
        print(f'ERROR: Gist PATCH failed for {filename}: {e}', file=sys.stderr)
        return False
    return True


base  = datetime.strptime(date_str, '%Y-%m-%d')
dates = [(base - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(backfill)]
ok = all([fill_date(d) for d in dates])
sys.exit(0 if ok else 1)
