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
    const t = tier(Math.abs(edge), cfg.spreadLean, cfg.spreadTake);
    if (t) {
      const home = edge > 0;
      picks.push({
        market: 'spread', tier: t,
        side: home ? g.home : g.away, sideIsHome: home,
        line: home ? g.vegasSpread.point : -g.vegasSpread.point,
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
        edge: Math.round(Math.abs(edge) * 10) / 10,
      });
    }
  }
  return picks;
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
  let written = 0, frozen = 0, unrated = 0;

  for (const g of games) {
    if (!g.time || now >= new Date(g.time).getTime() || g.statusName !== 'STATUS_SCHEDULED') {
      frozen++;
      continue;  // tipped (or postponed/cancelled): keep whatever was logged before
    }
    if (!g.pred) { unrated++; continue; }

    const prev = byId.get(g.id);
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
      picks: picksFor(g, cfg),
      result: null,
    };
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
}

module.exports = { picksFor };

if (require.main === module) {
  main().catch(e => { console.error(`ERROR: ${e.message}`); process.exit(1); });
}
