import json, urllib.request, os, sys
from datetime import datetime, timezone

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/loganthein/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'Note: could not fetch {filename}: {e}')
        return None

def gist_patch(filename, data):
    body = json.dumps({'files': {filename: {'content': json.dumps(data)}}}).encode()
    req = urllib.request.Request(
        f'https://api.github.com/gists/{gist_id}',
        data=body, method='PATCH',
        headers={
            'Authorization': f'token {gist_pat}',
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'lock-odds-workflow',
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status

live_raw = gist_fetch(f'odds-{date_str}.json')
if live_raw is None:
    print(f'ERROR: could not fetch odds-{date_str}.json — aborting lock step', file=sys.stderr)
    sys.exit(1)

live        = live_raw
locked_list = gist_fetch(f'locked-odds-{date_str}.json') or []
locked_ids  = {g['id'] for g in locked_list if 'id' in g}
now_ts      = datetime.now(timezone.utc).timestamp()
added       = 0

for game in live:
    gid = game.get('id')
    ct  = game.get('commence_time')
    if not gid or not ct:
        continue
    try:
        ct_ts = datetime.fromisoformat(ct.replace('Z', '+00:00')).timestamp()
    except Exception:
        continue
    if ct_ts <= now_ts and gid not in locked_ids:
        locked_list.append(game)
        locked_ids.add(gid)
        added += 1
        print(f'Locking: {game.get("home_team")} vs {game.get("away_team")} ({ct})')

if added == 0:
    print('No new games to lock.')
    sys.exit(0)

out = f'locked-odds-{date_str}.json'
print(f'Writing {out} with {len(locked_list)} games ({added} new)...')
try:
    status = gist_patch(out, locked_list)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
