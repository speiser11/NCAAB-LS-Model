"""
gist_api.py
Shared Gist write helper for the fetch-odds workflow.

Every script in this workflow PATCHes the same Gist, several times within a few
seconds of each other. GitHub occasionally rejects a write with 409 Conflict
when the previous write to that Gist's git ref hasn't settled yet — it is
transient, and the same payload succeeds on a retry. Each script used to call
urlopen once and exit(1) on any failure, which turned a one-second race into a
failed workflow run.

Callers pass content already serialized so per-script formatting (indent, raw
passthrough) is preserved.
"""

import json
import time
import urllib.error
import urllib.request

# 409 = concurrent write to the same Gist; 5xx = GitHub-side blip.
RETRY_STATUSES = {409, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 4, 8)


def patch(gist_id, gist_pat, filename, content, user_agent):
    """PATCH a single file into the Gist, retrying transient failures.

    `content` must already be a string. Returns the HTTP status on success;
    re-raises the final exception if every attempt fails.
    """
    body = json.dumps({'files': {filename: {'content': content}}}).encode()

    for attempt in range(len(BACKOFF_SECONDS) + 1):
        req = urllib.request.Request(
            f'https://api.github.com/gists/{gist_id}',
            data=body, method='PATCH',
            headers={
                'Authorization': f'token {gist_pat}',
                'Content-Type':  'application/json',
                'Accept':        'application/vnd.github.v3+json',
                'User-Agent':    user_agent,
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status
        except urllib.error.HTTPError as e:
            # Python clears the `as` binding when the handler exits, so keep
            # our own reference for the final re-raise.
            failure  = e
            retriable = e.code in RETRY_STATUSES
            reason    = f'HTTP {e.code}'
        except Exception as e:  # timeouts, connection resets, DNS blips
            failure   = e
            retriable = True
            reason    = str(e)

        if not retriable or attempt == len(BACKOFF_SECONDS):
            raise failure
        delay = BACKOFF_SECONDS[attempt]
        print(f'Gist PATCH {filename} failed ({reason}) — retrying in {delay}s')
        time.sleep(delay)
