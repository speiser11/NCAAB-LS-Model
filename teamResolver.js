// teamResolver.js
// Loads data/ncaab-team-aliases.json and exposes:
//   TeamResolver.getCanonical(rawName) → canonical string (or rawName if unknown)
//   TeamResolver.getDisplay(rawName)   → display string  (or rawName if unknown)
//   TeamResolver.init()                → Promise — call once at startup
//
// All functions are synchronous after init() resolves.
// Before init(), they return the raw input unchanged (safe no-op fallback).

(function () {
  // normalized alias → canonical name  (e.g. "duke blue devils" → "Duke")
  const aliasToCanonical = {};
  // normalized canonical → display name (e.g. "duke" → "Duke Blue Devils")
  const canonicalToDisplay = {};

  let ready = false;

  function norm(s) {
    return (s || '').toLowerCase().trim();
  }

  async function init() {
    if (ready) return;
    try {
      const res = await fetch('data/ncaab-team-aliases.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const entries = await res.json();
      for (const entry of entries) {
        const canonical = entry.canonical;
        const display   = entry.display || canonical;
        const normCanon = norm(canonical);
        canonicalToDisplay[normCanon] = display;
        // Map every alias (and the canonical itself) → canonical
        for (const alias of (entry.aliases || [])) {
          aliasToCanonical[norm(alias)] = canonical;
        }
        aliasToCanonical[normCanon] = canonical;
      }
      ready = true;
      console.info(`[TeamResolver] Loaded ${entries.length} team entries`);
    } catch (e) {
      console.warn('[TeamResolver] Failed to load aliases:', e.message);
    }
  }

  // Returns the canonical name for any alias.
  // Falls back to rawName if not found or not yet initialized.
  function getCanonical(rawName) {
    if (!rawName) return rawName;
    if (!ready)   return rawName;
    return aliasToCanonical[norm(rawName)] || rawName;
  }

  // Returns the display name (with mascot) for any alias.
  // Falls back to rawName if not found or not yet initialized.
  function getDisplay(rawName) {
    if (!rawName) return rawName;
    if (!ready)   return rawName;
    const canon = aliasToCanonical[norm(rawName)] || rawName;
    return canonicalToDisplay[norm(canon)] || canon;
  }

  window.TeamResolver = { init, getCanonical, getDisplay };
})();
