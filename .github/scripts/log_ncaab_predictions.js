/*
log_ncaab_predictions.js
Records the NCAAB model's pick for each game BEFORE tip-off, into
ncaab-log-YYYY-MM-DD.json on the odds Gist (date = US Eastern game date),
mirroring mlb-log-*.json. Results are filled in afterwards by
fill_ncaab_results.py.

Every run re-computes games that haven't tipped yet, so the stored pick is the
last one before tip-off (runs are 30 min apart). Once a game has started its
entry is never touched again — that frozen entry is what the record grades.

The prediction comes from index.html itself (see ncaab_site_model.js), so it
includes everything the site applies: Torvik ratings, home court, B2B, travel
and injuries. Lean/take tiers use the spread/total thresholds in
data/model-config.json — the same ones the site's game cards use.

Env: GIST_ID, GIST_PAT, DATE (optional YYYY-MM-DD; defaults to today, Eastern).
Run with TZ=America/New_York so the site code's local dates are Eastern.
*/

const path = require('path');
const site = require('./ncaab_site_model');

const GIST_USER = 'loganthein';
const gistId  = process.env.GIST_ID;
const gistPat = process.env.GIST_PAT;

function easternToday() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

async function gistFetch(filename) {
  const url = `https://gist.githubusercontent.com/${GIST_USER}/${gistId}/raw/${filename}`;
  const r = await fetch(url, { headers: { 'Cache-Control': 'no-cache' }, signal: AbortSignal.timeout(10000) });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Gist read ${filename}: HTTP ${r.status}`);
  return r.json();
}

// Mirrors gist_api.py: retry the transient 409/5xx a concurrent write can cause.
async function gistPatch(filename, data) {
  const body = JSON.stringify({ files: { [filename]: { content: JSON.stringify(data, null, 2) } } });
  for (const wait of [0, 2, 4, 8]) {
    if (wait) await new Promise(res => setTimeout(res, wait * 1000));
    const r = await fetch(`https://api.github.com/gists/${gistId}`, {
      method: 'PATCH', body,
      headers: { Authorization: `token ${gistPat}`, Accept: 'application/vnd.github+json',
                 'Content-Type': 'application/json', 'User-Agent': 'log-ncaab-predictions' },
    });
    if (r.ok) return r.status;
    if (![409, 500, 502, 503, 504].includes(r.status)) throw new Error(`Gist PATCH: HTTP ${r.status}`);
    console.log(`Gist PATCH ${r.status}, retrying…`);
  }
  throw new Error('Gist PATCH: retries exhausted');
}

function tier(absEdge, lean, take) {
  return absEdge >= take ? 'take' : absEdge >= lean ? 'lean' : null;
}

// The site's game-card advice, as data: spread edge = model margin vs the
// home line; total edge = model total vs the posted total.
function picksFor(g, cfg) {
  const picks = [];
  if (g.vegasSpread) {
    const edge = g.pred.margin + g.vegasSpread.point;
    // No spread pick on blowout lines (cfg.maxPickLine, e.g. 15+): untested there.
    // ...nor above cfg.spreadMaxEdge (backtest: past ~8 pts the gap is ratings noise).
    const t = Math.abs(g.vegasSpread.point) >= cfg.maxPickLine || Math.abs(edge) > cfg.spreadMaxEdge ? null
            : tier(Math.abs(edge), cfg.spreadLean, cfg.spreadTake);
    if (t) {
      const home = edge > 0;
      picks.push({
        market: 'spread', tier: t,
        side: home ? g.home : g.away, sideIsHome: home,
        line: home ? g.vegasSpread.point : -g.vegasSpread.point,
        price: (home ? g.vegasSpread.homePrice : g.vegasSpread.awayPrice) ?? null,   // American odds when posted
        edge: Math.round(Math.abs(edge) * 10) / 10,
      });
    }
  }
  if (g.vegasTotal) {
    const edge = g.pred.total - g.vegasTotal.point;
    const t = tier(Math.abs(edge), cfg.totalLean, cfg.totalTake);
    if (t) {
      picks.push({
        market: 'total', tier: t,
        side: edge > 0 ? 'over' : 'under',
        line: g.vegasTotal.point,
        price: (edge > 0 ? g.vegasTotal.overPrice : g.vegasTotal.underPrice) ?? null,
        edge: Math.round(Math.abs(edge) * 10) / 10,
      });
    }
  }
  return picks;
}

// Picks are posted, not recalculated. From POST_HOUR (US Eastern) on, the first
// time a game crosses the lean/take line its pick is posted with that line and
// time, and it stays even if the line moves — the line you could have bet is the
// one graded (fill_ncaab_results.py adds closing-line value). A lean whose edge
// grows into take range on the same side is upgraded at the new line; the
// original is kept in `was`. Returns the game's picks and what changed.
const POST_HOUR = 9;
function postPicks(prev, current, nowIso, open) {
  // Picks saved before posting existed (no posted_at) are re-derived, not kept.
  const picks = (prev || []).filter(p => p.posted_at);
  const changes = [];
  for (const c of current) {
    const i = picks.findIndex(p => p.market === c.market);
    if (i < 0) {
      if (!open) continue;
      picks.push({ ...c, posted_at: nowIso });
      changes.push({ kind: 'new', pick: picks[picks.length - 1] });
    } else if (open && picks[i].tier === 'lean' && c.tier === 'take' && picks[i].side === c.side) {
      const { tier, line, edge, posted_at } = picks[i];
      picks[i] = { ...c, posted_at: nowIso, was: { tier, line, edge, posted_at } };
      changes.push({ kind: 'upgrade', pick: picks[i] });
    }
  }
  return { picks, changes };
}

const pickLabel = p => p.market === 'spread'
  ? `${p.side} ${p.line > 0 ? '+' : ''}${p.line}`
  : `${p.side === 'over' ? 'Over' : 'Under'} ${p.line}`;
const etTime = iso => new Date(iso).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' });

const fmtPrice = n => n == null ? '' : ` (${n > 0 ? '+' : ''}${n})`;

// One push per run listing new takes and leans upgraded to takes (ntfy.sh topic in
// NTFY_TOPIC). New leans don't alert; they're on the site.
async function notify(changes) {
  const topic = process.env.NTFY_TOPIC;
  const takes = changes.filter(c => c.pick.tier === 'take');
  if (!topic || !takes.length) return;
  const body = takes.map(({ kind, pick, game, commence, range }) =>
    `${kind === 'upgrade' ? 'UPGRADED TO TAKE' : 'TAKE'}: ${pickLabel(pick)}${fmtPrice(pick.price)} | ${game}, ${etTime(commence)} ET (edge ${pick.edge})` +
    (range ? `\n   Still a bet at: ${range}` : '')).join('\n');
  try {
    const r = await fetch(`https://ntfy.sh/${encodeURIComponent(topic)}`, {
      method: 'POST', body,
      headers: { Title: `TrueLine: ${takes.length} new take${takes.length > 1 ? 's' : ''}`,
                 Tags: 'basketball', Priority: 'high', Click: 'https://truelinels.net' },
    });
    console.log(`Notification: HTTP ${r.status}`);
  } catch (e) {
    console.log(`Notification failed (picks are saved regardless): ${e.message}`);
  }
}

async function main() {
  const isoDate = process.env.DATE || easternToday();
  const espnDate = isoDate.replace(/-/g, '');
  const filename = `ncaab-log-${isoDate}.json`;

  const model = await site.load({ root: path.join(__dirname, '..', '..') });
  const { games, cfg, ratingsCount } = await model.run(espnDate);
  console.log(`${isoDate}: ${games.length} games on ESPN, ${ratingsCount} rated teams.`);
  if (!games.length) { console.log('No games — nothing to log.'); return; }

  const log = (await gistFetch(filename)) || [];
  const byId = new Map(log.map(e => [e.game_id, e]));
  const now = Date.now();
  const nowIso = new Date(now).toISOString();
  const open = new Date(now).getHours() >= POST_HOUR;   // TZ=America/New_York in the workflow
  const changes = [];
  let written = 0, frozen = 0, unrated = 0;

  for (const g of games) {
    if (!g.time || now >= new Date(g.time).getTime() || g.statusName !== 'STATUS_SCHEDULED') {
      frozen++;
      continue;  // tipped (or postponed/cancelled): keep whatever was logged before
    }
    if (!g.pred) { unrated++; continue; }

    const prev = byId.get(g.id);
    const posted = postPicks(prev?.picks, picksFor(g, cfg), nowIso, open);
    const entry = {
      game_id:   g.id,
      game:      `${g.away} @ ${g.home}`,
      home:      g.home,
      away:      g.away,
      hKey:      g.hKey,
      aKey:      g.aKey,
      neutral:   g.neutral,
      commence:  g.time,
      first_logged: prev?.first_logged || new Date(now).toISOString(),
      logged_at: new Date(now).toISOString(),
      model: {
        margin: g.pred.margin,      // + = home by this much
        total:  g.pred.total,
        homeWinProb: g.pred.homeWinProb,
        hB2B: g.hB2B, aB2B: g.aB2B, awayTravel: g.awayTravel,
        hInjuries: g.hInjuries.length, aInjuries: g.aInjuries.length,
        // Torvik games played (0 = preseason projection) — lets the record
        // separate early-season picks, when ratings are mostly priors.
        hGames: g.pred.hRating.games ?? null, aGames: g.pred.aRating.games ?? null,
      },
      vegas: {
        spread: g.vegasSpread?.point ?? null,  // home line (− = home favored)
        total:  g.vegasTotal?.point ?? null,
      },
      picks: posted.picks,
      result: null,
    };
    for (const c of posted.changes) {
      const p = c.pick;
      const range = model.betRange(g, { market: p.market, home: p.sideIsHome, over: p.side === 'over' });
      changes.push({ ...c, game: entry.game, commence: entry.commence, range });
    }
    if (prev && JSON.stringify({ ...prev, logged_at: 0 }) === JSON.stringify({ ...entry, logged_at: 0 })) continue;
    byId.set(g.id, entry);
    written++;
  }

  console.log(`Updated ${written}, already tipped ${frozen}, unrated (non-D1 opponent) ${unrated}.`);
  if (!written) return;

  const out = [...byId.values()].sort((a, b) => a.commence.localeCompare(b.commence));
  const nPicks = out.reduce((s, e) => s + e.picks.length, 0);
  console.log(`Writing ${filename}: ${out.length} games, ${nPicks} leans/takes.`);
  console.log(`Gist PATCH status: ${await gistPatch(filename, out)}`);
  if (!open) console.log(`Before ${POST_HOUR}am ET: games updated, no picks posted.`);
  console.log(`New/upgraded picks this run: ${changes.length} (${changes.filter(c => c.pick.tier === 'take').length} takes alerted)`);
  await notify(changes);
}

module.exports = { picksFor, postPicks };

if (require.main === module) {
  main().catch(e => { console.error(`ERROR: ${e.message}`); process.exit(1); });
}
