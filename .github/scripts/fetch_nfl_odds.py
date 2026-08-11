"""
fetch_nfl_odds.py
Fetches NFL odds (moneyline, spreads, totals) from The Odds API and pushes to Gist.
NFL games span multiple days per week (Thu/Sun/Mon), so unlike MLB's per-date files,
this writes a single evergreen nfl-odds.json covering all upcoming games. Games that
have already started are locked using the prior Gist snapshot before The Odds API
drops them from the response.
"""

import json, urllib.request, os, sys
from datetime import datetime, timezone

import gist_api

api_key  = os.environ['ODDS_API_KEY']
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']

GIST_USER = 'loganthein'
FILENAME  = 'nfl-odds.json'

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
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data), 'fetch-nfl-odds')

print('Fetching NFL odds...')
try:
    url = (
        'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
        f'?apiKey={api_key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'fetch-nfl-odds'})
    with urllib.request.urlopen(req, timeout=15) as r:
        remaining = r.headers.get('x-requests-remaining', '?')
        games = json.loads(r.read())
    print(f'Fetched {len(games)} games — credits remaining: {remaining}')
except Exception as e:
    print(f'NFL odds fetch failed (non-fatal, keeping existing Gist data): {e}')
    sys.exit(0)

if not games:
    print('No games returned — nothing to push.')
    sys.exit(0)

existing = gist_fetch(FILENAME) or []
existing_by_id = {eg['id']: eg for eg in existing if eg.get('id')}

# Lock odds for games that have already started (API drops in-progress lines).
now_utc = datetime.now(timezone.utc)
locked_count = 0
for game in games:
    ct = game.get('commence_time')
    if not ct:
        continue
    try:
        game_time = datetime.fromisoformat(ct.replace('Z', '+00:00'))
    except Exception:
        continue
    if game_time <= now_utc:
        prev = existing_by_id.get(game['id'])
        if prev and prev.get('bookmakers'):
            game['bookmakers'] = prev['bookmakers']
            locked_count += 1
if locked_count:
    print(f'Locked pre-game odds for {locked_count} started game(s).')

# Restore started games that dropped off the API response entirely.
games_by_id = {g['id']: g for g in games}
restored_count = 0
for eg in existing:
    if not eg.get('id') or eg['id'] in games_by_id:
        continue
    ct = eg.get('commence_time', '')
    try:
        game_time = datetime.fromisoformat(ct.replace('Z', '+00:00'))
    except Exception:
        continue
    if game_time <= now_utc:
        games.append(eg)
        restored_count += 1
if restored_count:
    print(f'Restored {restored_count} started game(s) from Gist cache.')

print(f'Pushing {FILENAME} to Gist...')
try:
    status = gist_patch(FILENAME, games)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
