"""
fetch_cfb_fpi.py
Pulls ESPN's Football Power Index for every FBS team and pushes it to the Gist
as cfb-fpi.json — the rating input behind the CFB model.

FPI is a neutral-field points-above-average rating, so the difference between
two teams' FPI plus a home-field constant projects a game's margin directly.
Checked against the 2026 week-1 board (60 games with both teams rated), FPI
explains 95% of the market's spread variance (r^2 = 0.95) with a mean absolute
error of 2.7 points once scaled — close enough to price games, far enough off to
disagree occasionally, which is the whole point.

FCS teams have no FPI. Those games are left unrated rather than guessed; the
front end shows Vegas only and makes no pick.
"""

import json, os, sys, urllib.request

import gist_api

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
season   = os.environ.get('SEASON', '2026')

FILENAME = 'cfb-fpi.json'
FPI_URL  = (
    'https://site.web.api.espn.com/apis/fitt/v3/sports/football/college-football'
    f'/powerindex?region=us&lang=en&limit=250&season={season}'
)


def gist_patch(filename, data):
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data), 'fetch-cfb-fpi')


print(f'Fetching FPI for {season}...')
try:
    req = urllib.request.Request(FPI_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.loads(r.read())
except Exception as e:
    print(f'FPI fetch failed (non-fatal, keeping existing Gist data): {e}')
    sys.exit(0)

teams = {}
for entry in d.get('teams', []):
    t = entry.get('team') or {}
    tid = t.get('id')
    cat = next((c for c in entry.get('categories', []) if c.get('name') == 'fpi'), None)
    vals = (cat or {}).get('values') or []
    if not tid or not vals:
        continue
    # values[] is positionally aligned with the category's names[]:
    # fpi, fpirank, rankchange7days, projectedw, projectedl, ...
    teams[str(tid)] = {
        'fpi':   round(float(vals[0]), 3),
        'name':  t.get('displayName', ''),
        'short': t.get('shortDisplayName', ''),
    }

if not teams:
    print('No FPI values returned — nothing to push (keeping existing data).')
    sys.exit(0)

payload = {
    'season': season,
    'updated': d.get('lastUpdated'),
    'teams': teams,
}
print(f'Parsed FPI for {len(teams)} teams.')

try:
    status = gist_patch(FILENAME, payload)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)

# ── Weekly snapshot, for backtesting ──────────────────────────────────────────
# The live file above is overwritten every run, so the ratings a pick was
# actually made from are gone within half an hour. Backtesting against
# end-of-season ratings instead is lookahead bias — the ratings would already
# know how the games turned out, and the model would look far better than it is.
#
# So keep one immutable snapshot per week, written the first time this runs in a
# new week and never touched again. It cannot be reconstructed after the fact,
# which is why it is worth doing before the analysis rather than after.
def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/loganthein/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


try:
    req = urllib.request.Request('https://cdn.espn.com/core/college-football/scoreboard?xhr=1',
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        sb = json.loads(r.read())
    sbd  = sb.get('content', {}).get('sbData', {})
    week = (sbd.get('week') or {}).get('number')
    styp = (sbd.get('season') or {}).get('type')
except Exception as e:
    week = styp = None
    print(f'Note: could not determine current week ({e}) — skipping snapshot.')

if week:
    snap = f'cfb-fpi-{season}-t{styp or 2}w{int(week):02d}.json'
    if gist_fetch(snap) is not None:
        print(f'Snapshot {snap} already exists — leaving it untouched.')
    else:
        try:
            print(f'Writing weekly snapshot {snap}...')
            print(f'Gist PATCH status: {gist_patch(snap, payload)}')
        except Exception as e:
            # Never let the snapshot break the live ratings write above.
            print(f'Note: snapshot write failed (non-fatal): {e}')
