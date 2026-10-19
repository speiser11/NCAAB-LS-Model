"""
fetch_alt_odds.py

Fetches MLB alternate spreads (-1.5 for both teams) once per day and
pushes to Gist as mlb-alt-odds-{date}.json.

Self-skips if today's file already exists in the Gist — safe to call
on every workflow run, only costs credits on the first run of the day.

alternate_spreads is an "additional market": the bulk /sports/{sport}/odds
endpoint rejects it with 422, so list the day's events (free) and request it
per-event, the same way fetch_mlb_odds.py fetches F5.

Env vars: ODDS_API_KEY, GIST_PAT, GIST_ID, DATE (optional, defaults to today UTC)
"""

import json, os, sys, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta

import gist_api

api_key  = os.environ['ODDS_API_KEY']
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')

GIST_USER = 'loganthein'
FILENAME  = f'mlb-alt-odds-{date_str}.json'

# ── Gist helpers (mirrors fetch_mlb_odds.py) ──────────────────────────────────
def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/{GIST_USER}/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None

def gist_patch(filename, data):
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data), 'fetch-alt-odds')

def odds_get(path, params):
    params['apiKey'] = api_key
    url = f'https://api.the-odds-api.com{path}?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'fetch-alt-odds'})
    with urllib.request.urlopen(req, timeout=20) as r:
        remaining = r.headers.get('x-requests-remaining', '?')
        return json.loads(r.read()), remaining

# ── Main ──────────────────────────────────────────────────────────────────────
existing = gist_fetch(FILENAME)
if existing is not None:
    print(f'Alt odds for {date_str} already in Gist ({len(existing)} games) — skipping.')
    sys.exit(0)

next_str = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

print(f'Fetching alternate_spreads for {date_str}...')
try:
    events, _ = odds_get('/v4/sports/baseball_mlb/events/', {
        'commenceTimeFrom': f'{date_str}T12:00:00Z',
        'commenceTimeTo':   f'{next_str}T12:00:00Z',
    })
    print(f'{len(events)} games')
except Exception as e:
    print(f'ERROR: events fetch failed: {e}', file=sys.stderr)
    sys.exit(1)

if not events:
    print('No games today — nothing to write.')
    sys.exit(0)

# ── Per event: find each team's -1.5 odds ────────────────────────────────────
output = []
failed = 0
for ev in events:
    home = ev.get('home_team', '')
    away = ev.get('away_team', '')
    home_m15 = None
    away_m15 = None

    try:
        g, remaining = odds_get(f'/v4/sports/baseball_mlb/events/{ev["id"]}/odds/', {
            'regions':    'us',
            'markets':    'alternate_spreads',
            'oddsFormat': 'american',
        })
    except Exception as e:
        print(f'  WARNING: alternate_spreads fetch failed for {away} @ {home}: {e}')
        failed += 1
        g, remaining = {}, '?'

    # Take the first bookmaker that quotes each side at -1.5
    for bk in g.get('bookmakers', []):
        alt = next((m for m in bk.get('markets', []) if m['key'] == 'alternate_spreads'), None)
        if not alt:
            continue
        for o in alt.get('outcomes', []):
            if o.get('point') == -1.5:
                if o.get('name') == home and home_m15 is None:
                    home_m15 = o.get('price')
                elif o.get('name') == away and away_m15 is None:
                    away_m15 = o.get('price')
        if home_m15 is not None and away_m15 is not None:
            break

    output.append({'home': home, 'away': away, 'home_m15': home_m15, 'away_m15': away_m15})
    print(f'  {away} @ {home}: home_m15={home_m15}, away_m15={away_m15} — credits remaining: {remaining}')

# Don't cache a day where every request failed — the next run would skip it.
if failed == len(events):
    print('ERROR: every per-event fetch failed; not writing Gist.', file=sys.stderr)
    sys.exit(1)

print(f'Pushing {FILENAME} to Gist...')
status = gist_patch(FILENAME, output)
print(f'Gist PATCH status: {status}')
