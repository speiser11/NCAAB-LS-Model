"""
fetch_alt_odds.py

Fetches MLB alternate spreads (-1.5 for both teams) once per day and
pushes to Gist as mlb-alt-odds-{date}.json.

Self-skips if today's file already exists in the Gist — safe to call
on every workflow run, only costs credits on the first run of the day.

Env vars: ODDS_API_KEY, GIST_PAT, GIST_ID, DATE (optional, defaults to today UTC)
"""

import json, os, sys, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta

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
    body = json.dumps({'files': {filename: {'content': json.dumps(data)}}}).encode()
    req  = urllib.request.Request(
        f'https://api.github.com/gists/{gist_id}',
        data=body, method='PATCH',
        headers={
            'Authorization': f'token {gist_pat}',
            'Content-Type':  'application/json',
            'Accept':        'application/vnd.github.v3+json',
            'User-Agent':    'fetch-alt-odds',
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status

def odds_get(params):
    params['apiKey'] = api_key
    url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?' + urllib.parse.urlencode(params)
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
    games, remaining = odds_get({
        'regions':          'us',
        'markets':          'alternate_spreads',
        'oddsFormat':       'american',
        'commenceTimeFrom': f'{date_str}T12:00:00Z',
        'commenceTimeTo':   f'{next_str}T12:00:00Z',
    })
    print(f'{len(games)} games — credits remaining: {remaining}')
except Exception as e:
    print(f'ERROR: alternate_spreads fetch failed: {e}', file=sys.stderr)
    sys.exit(1)

# ── Parse: for each game find each team's -1.5 odds ──────────────────────────
output = []
for g in games:
    home = g.get('home_team', '')
    away = g.get('away_team', '')
    home_m15 = None
    away_m15 = None

    for bk in g.get('bookmakers', []):
        alt = next((m for m in bk.get('markets', []) if m['key'] == 'alternate_spreads'), None)
        if not alt:
            continue
        for o in alt.get('outcomes', []):
            if o.get('point') == -1.5:
                if o.get('name') == home:
                    home_m15 = o.get('price')
                elif o.get('name') == away:
                    away_m15 = o.get('price')
        break  # first bookmaker is enough

    output.append({'home': home, 'away': away, 'home_m15': home_m15, 'away_m15': away_m15})
    print(f'  {away} @ {home}: home_m15={home_m15}, away_m15={away_m15}')

print(f'Pushing {FILENAME} to Gist...')
status = gist_patch(FILENAME, output)
print(f'Gist PATCH status: {status}')
