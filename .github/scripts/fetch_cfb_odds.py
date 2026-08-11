"""
fetch_cfb_odds.py
Fetches college football odds (moneyline, spreads, totals) from The Odds API
and pushes them to the Gist as a single evergreen cfb-odds.json.

Mirrors fetch_nfl_odds.py — CFB games span Tue through Sat, so per-date files
would fragment a single week across five files. Games that have already started
are locked from the prior Gist snapshot before The Odds API drops them.

One deliberate difference from the NFL script: the payload is trimmed to a
single bookmaker before it is stored. The front end only ever reads
bookmakers[0], so keeping all ~10 books multiplies the file size for data
nothing renders — and CFB has roughly four times as many games per week as the
NFL. Picking the book explicitly also stops the displayed line from flipping
between books run to run, which is what happens when you trust the API's
ordering.
"""

import json, urllib.request, os, sys
from datetime import datetime, timezone

import gist_api

api_key  = os.environ['ODDS_API_KEY']
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']

GIST_USER = 'loganthein'
FILENAME  = 'cfb-odds.json'

# First match wins; falls back to whatever the API returned if none are present.
BOOK_PREFERENCE = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers']


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
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data), 'fetch-cfb-odds')


def pick_book(game):
    """Reduce a game's bookmakers list to the single preferred book."""
    books = game.get('bookmakers') or []
    if not books:
        return game
    by_key = {b.get('key'): b for b in books}
    chosen = next((by_key[k] for k in BOOK_PREFERENCE if k in by_key), books[0])
    game['bookmakers'] = [chosen]
    return game


print('Fetching CFB odds...')
try:
    url = (
        'https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/'
        f'?apiKey={api_key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'fetch-cfb-odds'})
    with urllib.request.urlopen(req, timeout=20) as r:
        remaining = r.headers.get('x-requests-remaining', '?')
        games = json.loads(r.read())
    print(f'Fetched {len(games)} games — credits remaining: {remaining}')
except Exception as e:
    print(f'CFB odds fetch failed (non-fatal, keeping existing Gist data): {e}')
    sys.exit(0)

if not games:
    print('No games returned — nothing to push.')
    sys.exit(0)

games = [pick_book(g) for g in games]

existing = gist_fetch(FILENAME) or []
existing_by_id = {eg['id']: eg for eg in existing if eg.get('id')}

# Lock odds for games that have already kicked off (the API drops live lines).
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

# Drop games that finished long ago so the evergreen file cannot grow without
# bound across a season.
cutoff = now_utc.timestamp() - (14 * 86400)
before = len(games)
kept = []
for g in games:
    try:
        ts = datetime.fromisoformat(g.get('commence_time', '').replace('Z', '+00:00')).timestamp()
    except Exception:
        kept.append(g)          # keep anything unparseable rather than lose it
        continue
    if ts >= cutoff:
        kept.append(g)
games = kept
if before != len(games):
    print(f'Pruned {before - len(games)} game(s) older than 14 days.')

payload = json.dumps(games)
print(f'Pushing {FILENAME} to Gist ({len(games)} games, {len(payload):,} bytes)...')
try:
    status = gist_patch(FILENAME, games)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
