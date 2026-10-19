/*
ncaab_site_model.js
Runs the NCAAB model from index.html headlessly, so the workflow logs exactly
the prediction the site shows — no second copy of the math to drift out of step.

It lifts the needed declarations out of index.html by name (data loaders,
team matching, predict, assembleGames) and evaluates them in a sandbox with
small stand-ins for the browser APIs they touch (storage, status bar). If a
name below is renamed or removed in index.html, load() throws instead of
silently logging something different.

Usage:
  const site = await require('./ncaab_site_model').load({ root });
  const games = await site.run('20261103');   // ESPN date, YYYYMMDD
*/

const fs   = require('fs');
const path = require('path');
const vm   = require('vm');

// Declarations taken verbatim from index.html, in dependency order.
const NAMES = [
  'ESPN_URL', 'CONF_ID_MAP', 'CONF_TZ', 'ALIAS', 'cfg', 'NCAAB_SIGMA', 'FORM_WEIGHT',
  'GIST_URL', 'GIST_META_URL', 'GIST_RAW', 'ODDS_BASE',
  'fetchViaProxy', 'cacheAge', 'normTeam', 'matchTeam', 'espnIdToKey', 'loadEspnIds', 'espnTeamKey', 'loadModelConfig', 'loadTorvik', 'parseTorvik',
  '_cleanConfName', 'loadConferences', 'loadESPN', 'loadOdds', 'fetchStoredOdds',
  'gistToOddsFormat', 'loadRestData', 'loadTeamInjuries', 'loadInjuries',
  'normalCDF', 'injuryImpact', 'predict', 'blendRatings', 'getFormDelta', 'getLine',
  'assembleGames', 'todayStr', 'toISODate',
];

// Mutable page state the functions above read and write.
const STATE = `
let ratings = {}, espnList = [], oddsList = [], lockedOddsList = [];
let refData = {}, kenpomTeams = {}, bettingMap = {}, oddsLeft = null;
const _odds404Dates = new Set(), _oddsPendingDates = new Set();
let confIdMap = {}, teamIdToConf = {};  // espnIdToKey comes from index.html
let teamsPlayedYesterday = new Set(), teamDaysRest = {}, recentRatings = {}, injuryMap = {};
let currentIsoDate = '', _lastESPNRefresh = null;
`;

// Returns the source of one top-level declaration, matched by brace depth.
function extract(src, name) {
  const re = new RegExp(`^(?:async function|function|const|let) ${name.replace('$', '\\$')}\\b`, 'm');
  const m = re.exec(src);
  if (!m) throw new Error(`index.html: declaration "${name}" not found`);
  const start = m.index;
  const eol = src.indexOf('\n', start);
  const firstLine = src.slice(start, eol);
  // Single-line declaration, e.g. const ESPN_URL = '...';
  if (!/[{[]\s*$/.test(firstLine) && /;\s*(\/\/.*)?$/.test(firstLine)) return firstLine;
  let i;
  if (/^(const|let)/.test(firstLine)) {
    i = src.indexOf(firstLine.includes('= {') ? '{' : '[', start);
  } else {
    // Skip the parameter list first: it can hold braces ({ td, yd } = …)
    let p = src.indexOf('(', start), pd = 0;
    for (; p < src.length; p++) {
      if (src[p] === '(') pd++;
      else if (src[p] === ')' && --pd === 0) break;
    }
    i = src.indexOf('{', p);
  }
  let depth = 0;
  for (; i < src.length; i++) {
    const c = src[i];
    if (c === '{' || c === '[') depth++;
    else if (c === '}' || c === ']') { if (--depth === 0) break; }
  }
  let end = i + 1;
  if (src[end] === ';') end++;
  return src.slice(start, end);
}

async function load({ root }) {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const decls = NAMES.map(n => {
    const code = extract(html, n);
    try { new vm.Script(code); }
    catch (e) { throw new Error(`index.html: could not isolate "${n}": ${e.message}`); }
    return code;
  });

  const memStore = () => {
    const m = new Map();
    return {
      getItem: k => (m.has(k) ? m.get(k) : null),
      setItem: (k, v) => m.set(k, String(v)),
      removeItem: k => m.delete(k),
    };
  };

  // Relative URLs (data/model-config.json, data/ncaab-team-aliases.json) are
  // read from the checkout. The Odds API is never called from here: the site
  // only falls back to it when the centralized Gist is missing, and that
  // fallback must not spend credits from a workflow.
  const siteFetch = async (url, opts) => {
    if (!/^https?:/.test(url)) {
      const body = fs.readFileSync(path.join(root, url), 'utf8');
      return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
    }
    if (decodeURIComponent(url).includes('api.the-odds-api.com')) throw new Error('Odds API disabled in workflow');
    return fetch(url, opts);
  };

  const ctx = vm.createContext({
    fetch: siteFetch, Response, AbortSignal, URLSearchParams, console,
    setTimeout, clearTimeout, Promise, Date, Math, JSON,
    localStorage: memStore(), sessionStorage: memStore(),
    location: { hostname: 'workflow' },
    setSt: () => {},
    showErr: msg => console.warn(`[site] ${msg}`),
    window: {},
  });

  // TeamResolver (used by matchTeam) lives in teamResolver.js and attaches to window.
  vm.runInContext(fs.readFileSync(path.join(root, 'teamResolver.js'), 'utf8'), ctx);
  vm.runInContext('var TeamResolver = window.TeamResolver;', ctx);

  vm.runInContext(STATE + decls.join('\n\n') + `
    async function __run(dateStr) {
      currentIsoDate = toISODate(dateStr);
      await loadModelConfig();
      await Promise.all([
        TeamResolver.init(),
        loadEspnIds(),
        loadConferences(),
        loadTorvik(),
        loadESPN(dateStr),
        loadRestData(dateStr),
        loadOdds(currentIsoDate),
      ]);
      await loadInjuries();
      return { games: assembleGames(), cfg, ratingsCount: Object.keys(ratings).length };
    }
    globalThis.__run = __run;
    globalThis.__matchTeam = matchTeam;
    globalThis.__ratings = () => ratings;
  `, ctx);

  return {
    run: dateStr => ctx.__run(dateStr),
    matchTeam: name => ctx.__matchTeam(name),  // valid after run()
    ratings: () => ctx.__ratings(),
  };
}

module.exports = { load, extract, NAMES };
