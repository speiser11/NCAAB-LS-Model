"""
Collects last season's NCAAB opening and closing lines from BettingPros for the
backtest, into lines-2025-26.json. Runs from collect-lines.yml with the
ODDS_PROVIDER_KEY secret. Stops cleanly at MAX_CALLS so the daily plan (1,000
calls) keeps room for the live odds job; re-running resumes from the dates
not yet collected.
"""
import json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import date, timedelta
from pathlib import Path

KEY = os.environ['ODDS_PROVIDER_KEY']
MAX_CALLS = int(os.environ.get('MAX_CALLS', '450'))
SPREAD, TOTAL, CONSENSUS = 226, 225, 0
teams = json.loads(Path('data/ncaab-bettingpros-ids.json').read_text())
OUT = Path('lines-2025-26.json')
data = json.loads(OUT.read_text()) if OUT.exists() else {'dates_done': [], 'games': []}
calls = 0

def get(path, **q):
    global calls
    url = 'https://api.bettingpros.com/v3/' + path + '?' + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={'x-api-key': KEY, 'User-Agent': 'trueline-backtest'})
    for wait in (5, 10, 20, 40, None):
        if calls >= MAX_CALLS:
            raise RuntimeError('budget')
        time.sleep(1.0); calls += 1
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait is None: raise
            time.sleep(float(e.headers.get('Retry-After') or wait))

def get_all(path, arr, **q):
    items, page = [], 1
    while True:
        d = get(path, limit=50, page=page, **q)
        items += d.get(arr, [])
        if page >= (d.get('_pagination') or {}).get('total_pages', 1): return items
        page += 1

def consensus(sel):
    for b in sel.get('books', []):
        if b['id'] == CONSENSUS:
            for ln in b.get('lines', []):
                if ln.get('main') and ln.get('line') is not None: return ln['line']
    return None

# Every day of the 2025-26 season, opening night through the title game.
dates = [date(2025, 11, 3) + timedelta(days=i) for i in range((date(2026, 4, 6) - date(2025, 11, 3)).days + 1)]

try:
    for day in dates:
        ds = day.isoformat()
        if ds in data['dates_done']: continue
        evs = [e for e in get_all('events', 'events', sport='NCAAB', date=ds)
               if e.get('home') in teams and e.get('visitor') in teams]
        games = {e['id']: {'date': ds, 'scheduled': e['scheduled'], 'home': teams[e['home']]['torvik'],
                           'away': teams[e['visitor']]['torvik'], '_h': e['home']} for e in evs}
        ids = list(games)
        for i in range(0, len(ids), 25):
            for o in get_all('offers', 'offers', sport='NCAAB', market_id=f'{SPREAD}:{TOTAL}',
                             event_id=':'.join(map(str, ids[i:i + 25]))):
                g = games.get(o.get('event_id'))
                if not g: continue
                for s in o.get('selections', []):
                    close, open_ = consensus(s), (s.get('opening_line') or {}).get('line')
                    if o['market_id'] == SPREAD and s.get('participant') == g['_h']:
                        g['spread_close'], g['spread_open'] = close, open_      # home line
                    elif o['market_id'] == SPREAD and 'spread_close' not in g and s.get('participant'):
                        g['spread_close'] = -close if close is not None else None
                        g['spread_open'] = -open_ if open_ is not None else None
                    elif o['market_id'] == TOTAL and s.get('selection') == 'over':
                        g['total_close'], g['total_open'] = close, open_
        for g in games.values():
            g.pop('_h', None)
            if 'spread_close' in g or 'total_close' in g: data['games'].append(g)
        data['dates_done'].append(ds)
        print(f'{ds}: {len(games)} D1 games; total kept {len(data["games"])}; calls {calls}')
except RuntimeError:
    print(f'Stopped at the {MAX_CALLS}-call budget; re-run to continue.')
OUT.write_text(json.dumps(data))
print(f'DONE dates {len(data["dates_done"])}/{len(dates)}, games {len(data["games"])}, calls this run {calls}')
