"""
fetch_mlb_odds.py
Fetches MLB odds from The Odds API and pushes to Gist.
- Bulk endpoint: h2h, spreads, totals (every run)
- Per-event endpoint: h2h_h1, spreads_h1, totals_h1 (once per day — skipped if
  today's Gist file already has F5 data to conserve API credits)
"""

import json, urllib.request, urllib.parse, os, sys
from datetime import datetime, timezone

# ── Env ───────────────────────────────────────────────────────────────────────
api_key  = os.environ['ODDS_API_KEY']
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')

GIST_USER = 'loganthein'

# ── Gist helpers ──────────────────────────────────────────────────────────────
def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/{GIST_USER}/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'Note: could not fetch {filename}: {e}')
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
            'User-Agent':    'fetch-mlb-odds',
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status

def odds_get(path, params):
    params['apiKey'] = api_key
    url = f'https://api.the-odds-api.com{path}?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'fetch-mlb-odds'})
    with urllib.request.urlopen(req, timeout=15) as r:
        remaining = r.headers.get('x-requests-remaining', '?')
        data = json.loads(r.read())
        return data, remaining

# ── Step 1: Fetch bulk odds (h2h, spreads, totals) ───────────────────────────
date_next = datetime.strptime(date_str, '%Y-%m-%d')
from datetime import timedelta
next_str = (date_next + timedelta(days=1)).strftime('%Y-%m-%d')

print('Fetching bulk MLB odds...')
try:
    games, remaining = odds_get('/v4/sports/baseball_mlb/odds/', {
        'regions':           'us',
        'markets':           'h2h,spreads,totals',
        'oddsFormat':        'american',
        'commenceTimeFrom':  f'{date_str}T12:00:00Z',
        'commenceTimeTo':    f'{next_str}T12:00:00Z',
    })
    print(f'Bulk: {len(games)} games — credits remaining: {remaining}')
except Exception as e:
    print(f'ERROR: bulk odds fetch failed: {e}', file=sys.stderr)
    sys.exit(1)

if not games:
    print('No games returned — nothing to push.')
    sys.exit(0)

# Build lookup by event ID for easy merging
games_by_id = {g['id']: g for g in games}

# ── Step 2: Check if F5 already fetched today ────────────────────────────────
existing = gist_fetch(f'mlb-odds-{date_str}.json') or []
f5_already_fetched = False
if existing:
    # Check if any game already has F5 market data
    for eg in existing:
        mkts = eg.get('bookmakers', [{}])[0].get('markets', []) if eg.get('bookmakers') else []
        if any(m['key'] in ('h2h_h1', 'spreads_h1', 'totals_h1') for m in mkts):
            f5_already_fetched = True
            break

if f5_already_fetched:
    print('F5 data already present in Gist — skipping per-event fetch to conserve credits.')
    # Restore F5 markets from existing data into current games
    existing_by_id = {eg['id']: eg for eg in existing}
    for game in games:
        prev = existing_by_id.get(game['id'])
        if not prev:
            continue
        prev_mkts = prev.get('bookmakers', [{}])[0].get('markets', []) if prev.get('bookmakers') else []
        f5_mkts = [m for m in prev_mkts if m['key'] in ('h2h_h1', 'spreads_h1', 'totals_h1')]
        if f5_mkts and game.get('bookmakers'):
            game['bookmakers'][0]['markets'].extend(f5_mkts)
else:
    # ── Step 3: Fetch F5 per-event ────────────────────────────────────────────
    print(f'Fetching F5 odds for {len(games)} games...')
    for game in games:
        event_id = game['id']
        try:
            f5_data, remaining = odds_get(
                f'/v4/sports/baseball_mlb/events/{event_id}/odds/', {
                    'regions':    'us',
                    'markets':    'h2h_h1,spreads_h1,totals_h1',
                    'oddsFormat': 'american',
                }
            )
            # Merge F5 markets into the game's first bookmaker
            f5_bookmakers = f5_data.get('bookmakers', [])
            if f5_bookmakers and game.get('bookmakers'):
                f5_mkts = [m for m in f5_bookmakers[0].get('markets', [])
                           if m['key'] in ('h2h_h1', 'spreads_h1', 'totals_h1')]
                game['bookmakers'][0]['markets'].extend(f5_mkts)
                print(f'  {game["away_team"]} @ {game["home_team"]}: '
                      f'{len(f5_mkts)} F5 markets — credits remaining: {remaining}')
            else:
                print(f'  {game["away_team"]} @ {game["home_team"]}: no F5 bookmakers')
        except Exception as e:
            print(f'  WARNING: F5 fetch failed for {game.get("home_team")}: {e}')

# ── Step 4: Push merged data to Gist ─────────────────────────────────────────
filename = f'mlb-odds-{date_str}.json'
print(f'Pushing {filename} to Gist...')
try:
    status = gist_patch(filename, games)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
