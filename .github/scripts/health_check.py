"""
health_check.py
Last step of fetch-odds.yml. Every other step runs with continue-on-error, so a
one-off blip (e.g. GitHub's Gist API returning 500 for a few seconds) no longer
fails the run and emails everyone. This step decides whether to fail instead:

- It keeps a failure streak in fetch-health.json on the odds Gist.
- A run where any step failed adds one; a clean run resets it to 0.
- The run is marked failed (GitHub emails) when the streak reaches ALERT_AFTER,
  then every REMIND_EVERY failed runs while it lasts, not on every run.

If the health file itself can't be read or written (the Gist is down), this
warns and passes: that is the same kind of blip we're filtering out, and a
lasting Gist outage is GitHub-wide and shows up on their status page.

Env: GIST_ID, GIST_PAT, OUTCOMES (space-separated step outcomes from the workflow).
"""

import json, os, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

import gist_api

ALERT_AFTER  = 3   # failed runs in a row before the first email (45-90 min at 15-30 min cadence)
REMIND_EVERY = 8   # then one more email every this many failed runs
FILE = 'fetch-health.json'

gist_id, gist_pat = os.environ['GIST_ID'], os.environ['GIST_PAT']
outcomes = os.environ.get('OUTCOMES', '').split()
failed = 'failure' in outcomes
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def read_health():
    url = f'https://gist.githubusercontent.com/loganthein/{gist_id}/raw/{FILE}?t={int(time.time())}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}), timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'streak': 0}          # first run: no file yet
        raise

try:
    prev = read_health()
except Exception as e:
    print(f'::warning::Could not read {FILE} ({e}); not failing the run.')
    sys.exit(0)

streak = prev.get('streak', 0) + 1 if failed else 0
print(f'Step outcomes: {" ".join(outcomes) or "none"}')
print(f'Failed runs in a row: {streak}')

if streak != prev.get('streak', 0):      # only write when it changes (a clean run after a clean run writes nothing)
    health = {'streak': streak, 'updated_at': now,
              'failing_since': (prev.get('failing_since') or now) if failed else None}
    try:
        gist_api.patch(gist_id, gist_pat, FILE, json.dumps(health), 'fetch-health')
    except Exception as e:
        print(f'::warning::Could not write {FILE} ({e}); not failing the run.')
        sys.exit(0)

if failed and streak >= ALERT_AFTER and (streak - ALERT_AFTER) % REMIND_EVERY == 0:
    print(f'::error::Fetch Odds has had errors {streak} runs in a row '
          f'(since {prev.get("failing_since") or now}). Check the failed steps above.')
    sys.exit(1)
if failed:
    print(f'::warning::A step failed this run ({streak} in a row); '
          f'emailing only after {ALERT_AFTER} in a row.')
