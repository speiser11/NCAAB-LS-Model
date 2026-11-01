"""
fetch_bettingpros_odds.py
Fetches NCAAB spreads and totals for one day from BettingPros and pushes them to
the odds Gist as odds-{DATE}.json — the file index.html and the pick log read.

The output keeps the shape the site already parses (it was The Odds API's), one
bookmaker: BettingPros' consensus line, so nothing downstream changes:
  {id, commence_time, home_team, away_team,
   bookmakers: [{key, title, markets: [{key: 'spreads'|'totals', outcomes: [...]}]}]}

Teams are matched by BettingPros team id through data/ncaab-bettingpros-ids.json
(all 365 D1 teams, audited 1:1 against Torvik) and written under their exact
Torvik names, which the site matches exactly. Matching on BettingPros' names
would pair "Michigan St." with Michigan. Games with a team outside that table
(non-D1 opponents) are skipped — they have no ratings either.

Env: ODDS_PROVIDER_KEY (BettingPros API key), GIST_ID, GIST_PAT,
     DATE (YYYY-MM-DD, the US Eastern game date).
     --out FILE writes the result to FILE instead of the Gist.
"""

import argparse, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import gist_api

BASE = 'https://api.bettingpros.com/v3'
SPREAD, TOTAL = 226, 225          # NCAAB market ids (GET /v3/markets?sport=NCAAB)
CONSENSUS = 0                     # book id of "BettingPros Consensus"
# If a game has no consensus line yet, take the first of these books that has one.
FALLBACK_BOOKS = [12, 10, 13, 19, 49, 14, 18, 33, 24]  # DK, FD, Caesars, MGM, Hard Rock, Fanatics, BetRivers, theScore, bet365

ap = argparse.ArgumentParser()
ap.add_argument('--out', help='write to this file instead of the Gist')
args = ap.parse_args()

key      = os.environ['ODDS_PROVIDER_KEY']
date_str = os.environ['DATE']
teams    = json.loads((Path(__file__).resolve().parents[2] / 'data' / 'ncaab-bettingpros-ids.json').read_text())

PACE = 1.0                        # seconds between calls; BettingPros rate-limits bursts (429)
calls = 0                         # the plan allows 1,000 calls a day; each run prints its count

def get(path, **q):
    """GET with pacing; on 429 waits (Retry-After, else 5/10/20/40s) and retries."""
    url = f'{BASE}/{path}?' + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={'x-api-key': key, 'User-Agent': 'trueline-fetch-odds'})
    global calls
    for wait in (5, 10, 20, 40, None):
        time.sleep(PACE)
        calls += 1
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait is None:
                raise
            delay = float(e.headers.get('Retry-After') or wait)
            print(f'  429 from /{path}, retrying in {delay:.0f}s')
            time.sleep(delay)

def get_all(path, array, **q):
    """Every item across pages (BettingPros pages at 50)."""
    items, page = [], 1
    while True:
        d = get(path, limit=50, page=page, **q)
        items += d.get(array, [])
        if page >= (d.get('_pagination') or {}).get('total_pages', 1):
            return items
        page += 1

def main_line(selection):
    """(line, cost) from the consensus book, else the first fallback book; None if neither."""
    books = {b['id']: b for b in selection.get('books', [])}
    for book_id in [CONSENSUS] + FALLBACK_BOOKS:
        for ln in (books.get(book_id) or {}).get('lines', []):
            if ln.get('main') and ln.get('active') and not ln.get('is_off') and ln.get('line') is not None:
                return ln['line'], ln.get('cost')
    return None

try:
    events = get_all('events', 'events', sport='NCAAB', date=date_str)
except urllib.error.URLError as e:
    print(f'BettingPros events fetch failed for {date_str}: {e} — keeping existing Gist data')
    sys.exit(0)

games = {}
for ev in events:
    home, away = teams.get(ev.get('home')), teams.get(ev.get('visitor'))
    if not home or not away:
        continue
    games[ev['id']] = {
        'id': f"bp-{ev['id']}",
        'commence_time': ev['scheduled'].replace(' ', 'T') + 'Z',   # BettingPros times are UTC
        'home_team': home['torvik'],
        'away_team': away['torvik'],
        '_home_bp': ev['home'],
        'markets': {},
    }
print(f'{date_str}: {len(events)} NCAAB events, {len(games)} with both teams D1')

# Lines only for games that haven't started: a started game's line is already
# frozen in the pick log and locked-odds, and skipping them saves calls.
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
ids = [i for i, g in games.items() if g['commence_time'] > now]
try:
    for i in range(0, len(ids), 25):     # 25 events × 2 markets = one 50-offer page
        chunk = ':'.join(str(x) for x in ids[i:i + 25])
        for offer in get_all('offers', 'offers', sport='NCAAB', market_id=f'{SPREAD}:{TOTAL}', event_id=chunk):
            g = games.get(offer.get('event_id'))
            if not g or not offer.get('active'):
                continue
            outcomes = []
            for sel in offer.get('selections', []):
                ml = main_line(sel)
                if not ml:
                    continue
                if offer['market_id'] == SPREAD:
                    side = sel.get('participant')
                    name = g['home_team'] if side == g['_home_bp'] else g['away_team']
                    outcomes.append({'name': name, 'point': ml[0], 'price': ml[1]})
                elif sel.get('selection') in ('over', 'under'):
                    outcomes.append({'name': sel['selection'].title(), 'point': ml[0], 'price': ml[1]})
            if len(outcomes) == 2:
                g['markets']['spreads' if offer['market_id'] == SPREAD else 'totals'] = outcomes
except urllib.error.URLError as e:
    print(f'BettingPros offers fetch failed for {date_str}: {e} — keeping existing Gist data')
    sys.exit(0)

# Started games keep the entry from the last run (the line just before tip-off),
# read from the Gist rather than spending calls on them.
def previous_file():
    if args.out:
        return []
    url = f'https://gist.githubusercontent.com/loganthein/{os.environ["GIST_ID"]}/raw/odds-{date_str}.json?t={int(time.time())}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}), timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return []

fresh = {f"bp-{i}" for i in ids}
out = [g for g in previous_file() if g.get('id') not in fresh and g.get('commence_time', '') <= now]
for g in sorted(games.values(), key=lambda g: g['commence_time']):
    if not g['markets']:
        continue                          # no lines posted yet
    out.append({
        'id': g['id'], 'commence_time': g['commence_time'],
        'home_team': g['home_team'], 'away_team': g['away_team'],
        'bookmakers': [{
            'key': 'bettingpros_consensus', 'title': 'BettingPros Consensus',
            'markets': [{'key': k, 'outcomes': v} for k, v in g['markets'].items()],
        }],
    })
n_sp = sum(1 for g in out if any(m['key'] == 'spreads' for m in g['bookmakers'][0]['markets']))
n_to = sum(1 for g in out if any(m['key'] == 'totals' for m in g['bookmakers'][0]['markets']))
out.sort(key=lambda g: g['commence_time'])
print(f'{date_str}: {len(out)} games with lines ({n_sp} spreads, {n_to} totals); '
      f'{len(games) - len(ids)} already started; {calls} BettingPros calls')

content = json.dumps(out)
if args.out:
    Path(args.out).write_text(content)
    print(f'Wrote {args.out}')
else:
    print(f'Gist PATCH odds-{date_str}.json: '
          f'{gist_api.patch(os.environ["GIST_ID"], os.environ["GIST_PAT"], f"odds-{date_str}.json", content, "fetch-bettingpros-odds")}')
