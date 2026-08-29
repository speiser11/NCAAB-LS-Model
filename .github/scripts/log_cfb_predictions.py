"""
log_cfb_predictions.py
Records what the CFB model projected for each game BEFORE kickoff, into
cfb-log-YYYY-MM-DD.json on the Gist — one file per game date, mirroring
mlb-log-*.json.

This is the half of a model that cannot be reconstructed later: once a game
kicks off the pre-game line is gone from the API, and without a stored
projection there is no way to say whether the model was right. Results are
filled in afterwards by fill_cfb_results.py.

An entry is written once and then left alone, so the line recorded is the one
that stood the first time the model saw the game, not a number that drifts with
the market until kickoff.

Projection matches the front end: scaled FPI difference plus home field. Keep
MODEL_* below in step with cfb* in data/model-config.json.
"""

import json, os, re, sys, urllib.request
from datetime import datetime, timezone

import gist_api

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']

GIST_USER = 'loganthein'
ESPN_URL  = ('https://cdn.espn.com/core/college-football/schedule'
             '?xhr=1&year={year}&week={week}&group=80')

MODEL_SCALE     = 1.05
MODEL_HFA       = 3.5
MODEL_TAKE_EDGE = 6.0
MODEL_LEAN_EDGE = 3.5
MODEL_MAX_EDGE  = 10.0


import unicodedata

# Same school aliases and St/State handling the front end uses, so a game the
# site shows a pick for is a game that gets logged. If these drift apart the log
# stops being a record of what was actually published.
SCHOOL_ALIASES = [
    (r'\bnc state\b', 'north carolina state'), (r'\bn c state\b', 'north carolina state'),
    (r'\bpitt\b', 'pittsburgh'), (r'\bumass\b', 'massachusetts'),
    (r'\bsouthern miss\b', 'southern mississippi'), (r'\bul monroe\b', 'louisiana monroe'),
    (r'\bulm\b', 'louisiana monroe'), (r'\bfiu\b', 'florida international'),
    (r'\bfau\b', 'florida atlantic'),
]


def _base(s):
    s = unicodedata.normalize('NFD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', s)).strip()


def norm(s):
    return _base(s).replace(' ', '')


def variants(name):
    base = _base(name)
    if not base:
        return []
    canon = base
    for pat, repl in SCHOOL_ALIASES:
        canon = re.sub(pat, repl, canon)
    out = set()
    for form in {base, canon}:
        out.add(form.replace(' ', ''))
        out.add(re.sub(r'\bst\b', 'state', form).replace(' ', ''))
        out.add(re.sub(r'\bstate\b', 'st', form).replace(' ', ''))
    return [v for v in out if v]


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
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data, indent=2), 'log-cfb-predictions')


def espn_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


# ── Inputs ────────────────────────────────────────────────────────────────────
fpi_doc = gist_fetch('cfb-fpi.json') or {}
fpi = fpi_doc.get('teams') or {}
if not fpi:
    print('No FPI ratings available — cannot project, nothing logged.')
    sys.exit(0)

odds_list = gist_fetch('cfb-odds.json') or []
if not odds_list:
    print('No CFB odds available — nothing to log.')
    sys.exit(0)

# Vegas home spread and moneylines, keyed both ways round for name drift.
vegas = {}
for g in odds_list:
    bk = (g.get('bookmakers') or [{}])[0]
    markets = bk.get('markets', [])
    spreads = next((m for m in markets if m.get('key') == 'spreads'), None)
    h2h     = next((m for m in markets if m.get('key') == 'h2h'), None)
    totals  = next((m for m in markets if m.get('key') == 'totals'), None)
    if not spreads:
        continue
    home_sp = next((o.get('point') for o in spreads.get('outcomes', []) if o.get('name') == g.get('home_team')), None)
    away_sp = next((o.get('point') for o in spreads.get('outcomes', []) if o.get('name') == g.get('away_team')), None)
    if home_sp is None:
        continue
    over = next((o for o in (totals or {}).get('outcomes', []) if o.get('name') == 'Over'), None)
    entry = {
        'homeSpread': home_sp,
        'awaySpread': away_sp,
        'homeML': next((o.get('price') for o in (h2h or {}).get('outcomes', []) if o.get('name') == g.get('home_team')), None),
        'awayML': next((o.get('price') for o in (h2h or {}).get('outcomes', []) if o.get('name') == g.get('away_team')), None),
        'totalLine': (over or {}).get('point'),
        'book': bk.get('title'),
    }
    for hk in variants(g.get('home_team')):
        for ak in variants(g.get('away_team')):
            vegas[(hk, ak)] = entry

# Which week to log. ESPN's scoreboard tells us the current one.
try:
    sb = espn_get('https://cdn.espn.com/core/college-football/scoreboard?xhr=1')
    season = sb.get('content', {}).get('sbData', {}).get('season', {}) or {}
    wk     = sb.get('content', {}).get('sbData', {}).get('week', {}) or {}
    year, week = season.get('year'), wk.get('number')
except Exception:
    year = week = None
year = int(os.environ.get('SEASON') or year or datetime.now(timezone.utc).year)
week = int(os.environ.get('WEEK') or week or 1)
print(f'Logging CFB predictions for {year} week {week}...')

try:
    sched = espn_get(ESPN_URL.format(year=year, week=week))
except Exception as e:
    print(f'ESPN schedule fetch failed: {e}', file=sys.stderr)
    sys.exit(0)

events = []
for _day, blk in (sched.get('content', {}).get('schedule', {}) or {}).items():
    events += blk.get('games', [])
print(f'{len(events)} games on the board.')

# ── Build entries, grouped by the game's UTC date ─────────────────────────────
now = datetime.now(timezone.utc)
by_date = {}
skipped_started = skipped_unrated = skipped_noline = 0

for ev in events:
    comp = (ev.get('competitions') or [{}])[0]
    cs   = comp.get('competitors') or []
    home = next((c for c in cs if c.get('homeAway') == 'home'), None)
    away = next((c for c in cs if c.get('homeAway') == 'away'), None)
    if not home or not away:
        continue
    kickoff = ev.get('date') or ''
    try:
        kt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
    except Exception:
        continue
    if kt <= now:
        skipped_started += 1
        continue          # never log a projection for a game already under way

    h_name = home.get('team', {}).get('displayName', '')
    a_name = away.get('team', {}).get('displayName', '')
    hf = fpi.get(str(home.get('team', {}).get('id')))
    af = fpi.get(str(away.get('team', {}).get('id')))
    if not hf or not af:
        skipped_unrated += 1
        continue

    v = next((vegas[(hk, ak)]
              for hk in variants(h_name) for ak in variants(a_name)
              if (hk, ak) in vegas), None)
    if not v:
        skipped_noline += 1
        continue

    margin      = MODEL_SCALE * (hf['fpi'] - af['fpi']) + MODEL_HFA
    home_spread = -margin
    edge        = home_spread - v['homeSpread']
    abs_edge    = abs(edge)
    suspect     = abs_edge > MODEL_MAX_EDGE
    tier = None if suspect else ('take' if abs_edge >= MODEL_TAKE_EDGE
                                 else 'lean' if abs_edge >= MODEL_LEAN_EDGE else None)
    take_home = edge < 0
    pick = None
    if tier:
        pick = {
            'side': h_name if take_home else a_name,
            'line': v['homeSpread'] if take_home else v['awaySpread'],
            'edge': round(abs_edge, 2),
        }

    date_key = kt.strftime('%Y-%m-%d')
    by_date.setdefault(date_key, []).append({
        'id':       ev.get('id'),
        'kickoff':  kickoff,
        'home':     h_name,
        'away':     a_name,
        'homeId':   str(home.get('team', {}).get('id')),
        'awayId':   str(away.get('team', {}).get('id')),
        'model': {
            'homeFpi':     hf['fpi'],
            'awayFpi':     af['fpi'],
            'homeSpread':  round(home_spread, 2),
            'margin':      round(margin, 2),
            'edge':        round(edge, 2),
            'tier':        tier,
            'suspect':     suspect,
            'pick':        pick,
        },
        'vegas':  v,
        'loggedAt': now.isoformat(timespec='seconds'),
        'result':  None,
    })

print(f'skipped — already started: {skipped_started}, unrated: {skipped_unrated}, no line: {skipped_noline}')

if not by_date:
    print('Nothing to log.')
    sys.exit(0)

# ── Write one file per date, only adding games not already recorded ───────────
total_new = 0
for date_key, entries in sorted(by_date.items()):
    filename = f'cfb-log-{date_key}.json'
    existing = gist_fetch(filename) or []
    seen = {e.get('id') for e in existing if e.get('id')}
    new = [e for e in entries if e['id'] not in seen]
    if not new:
        print(f'{date_key}: all {len(entries)} games already logged.')
        continue
    merged = existing + new
    merged.sort(key=lambda e: e.get('kickoff') or '')
    try:
        status = gist_patch(filename, merged)
        picks = sum(1 for e in new if e['model']['tier'])
        print(f'{date_key}: logged {len(new)} new game(s), {picks} with a pick '
              f'({len(merged)} total) — PATCH {status}')
        total_new += len(new)
    except Exception as e:
        print(f'ERROR: Gist PATCH failed for {filename}: {e}', file=sys.stderr)
        sys.exit(1)

print(f'Done — {total_new} new prediction(s) logged.')
