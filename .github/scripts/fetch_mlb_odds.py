"""
fetch_mlb_odds.py
Fetches MLB odds from The Odds API and pushes to Gist.
- Bulk endpoint: tries all 6 markets including F5 (correct MLB key names)
- Falls back to bulk without F5 + per-event with F5 keys if bulk rejects them
- F5 correct keys: h2h_1st_5_innings, spreads_1st_5_innings, totals_1st_5_innings
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

F5_MARKETS = 'h2h_1st_5_innings,spreads_1st_5_innings,totals_1st_5_innings'
ALL_MARKETS = f'h2h,spreads,totals,{F5_MARKETS}'

print('Fetching bulk MLB odds (including F5)...')
f5_in_bulk = False
try:
    games, remaining = odds_get('/v4/sports/baseball_mlb/odds/', {
        'regions':           'us',
        'markets':           ALL_MARKETS,
        'oddsFormat':        'american',
        'commenceTimeFrom':  f'{date_str}T12:00:00Z',
        'commenceTimeTo':    f'{next_str}T12:00:00Z',
    })
    f5_in_bulk = True
    print(f'Bulk (with F5): {len(games)} games — credits remaining: {remaining}')
except Exception as e:
    # Bulk with F5 failed — retry with base markets only, then fetch F5 per-event
    print(f'Bulk with F5 failed ({e}), retrying with base markets only...')
    f5_in_bulk = False
    try:
        games, remaining = odds_get('/v4/sports/baseball_mlb/odds/', {
            'regions':           'us',
            'markets':           'h2h,spreads,totals',
            'oddsFormat':        'american',
            'commenceTimeFrom':  f'{date_str}T12:00:00Z',
            'commenceTimeTo':    f'{next_str}T12:00:00Z',
        })
        print(f'Bulk (base only): {len(games)} games — credits remaining: {remaining}')
    except Exception as e2:
        print(f'ERROR: bulk odds fetch failed: {e2}', file=sys.stderr)
        sys.exit(1)

if not games:
    print('No games returned — nothing to push.')
    sys.exit(0)

# ── Fetch existing Gist data (used for F5 cache + locking started games) ──────
existing = gist_fetch(f'mlb-odds-{date_str}.json') or []
existing_by_id = {eg['id']: eg for eg in existing if eg.get('id')}

# ── Step 2: F5 per-event if not already in bulk ───────────────────────────────
if not f5_in_bulk:
    f5_already_fetched = any(
        any(m['key'] in ('h2h_1st_5_innings', 'spreads_1st_5_innings', 'totals_1st_5_innings')
            for m in (eg.get('bookmakers', [{}])[0].get('markets', []) if eg.get('bookmakers') else []))
        for eg in existing
    )

    if f5_already_fetched:
        print('F5 data already present in Gist — restoring from cache.')
        for game in games:
            prev = existing_by_id.get(game['id'])
            if not prev or not game.get('bookmakers'):
                continue
            prev_mkts = prev.get('bookmakers', [{}])[0].get('markets', []) if prev.get('bookmakers') else []
            f5_mkts = [m for m in prev_mkts
                       if m['key'] in ('h2h_1st_5_innings', 'spreads_1st_5_innings', 'totals_1st_5_innings')]
            if f5_mkts:
                game['bookmakers'][0]['markets'].extend(f5_mkts)
    else:
        print(f'Fetching F5 odds per-event for {len(games)} games...')
        for game in games:
            event_id = game['id']
            try:
                f5_data, remaining = odds_get(
                    f'/v4/sports/baseball_mlb/events/{event_id}/odds/', {
                        'regions':    'us',
                        'markets':    F5_MARKETS,
                        'oddsFormat': 'american',
                    }
                )
                f5_bookmakers = f5_data.get('bookmakers', [])
                if f5_bookmakers and game.get('bookmakers'):
                    f5_mkts = [m for m in f5_bookmakers[0].get('markets', [])
                               if m['key'] in ('h2h_1st_5_innings', 'spreads_1st_5_innings', 'totals_1st_5_innings')]
                    game['bookmakers'][0]['markets'].extend(f5_mkts)
                    print(f'  {game["away_team"]} @ {game["home_team"]}: '
                          f'{len(f5_mkts)} F5 markets — credits remaining: {remaining}')
                else:
                    print(f'  {game["away_team"]} @ {game["home_team"]}: no F5 bookmakers yet')
            except Exception as e:
                print(f'  WARNING: F5 fetch failed for {game.get("home_team")}: {e}')

# ── Step 3: Lock odds for games that have already started ────────────────────
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
            print(f'  Locked: {game.get("away_team")} @ {game.get("home_team")} (game started)')
if locked_count:
    print(f'Locked pre-game odds for {locked_count} started game(s).')

# ── Step 3b: Re-add started games that dropped off the API response ───────────
# The Odds API stops returning games once they start, so we must restore them
# from the existing Gist to avoid losing their locked pre-game odds.
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
        print(f'  Restored: {eg.get("away_team")} @ {eg.get("home_team")} (started, dropped from API)')
if restored_count:
    print(f'Restored {restored_count} started game(s) from Gist cache.')

# ── Step 4: Push merged data to Gist ─────────────────────────────────────────
filename = f'mlb-odds-{date_str}.json'
print(f'Pushing {filename} to Gist...')
try:
    status = gist_patch(filename, games)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
