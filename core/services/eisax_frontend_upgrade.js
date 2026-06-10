/**
 * eisax_frontend_upgrade.js
 * ─────────────────────────
 * Frontend patches for app.js — adds:
 *   1. GCC Intelligence Section (TASI, DFM, Oil link, headlines)
 *   2. Cross-Asset Signals banner (divergences)
 *   3. News Attribution footer row
 *   4. Full Report upgrade (news-grounded narrative)
 *
 * HOW TO USE:
 *   - Add the HTML sections to app.js's view-daily section
 *   - Add the JS functions below
 *   - Call them from renderDaily() and renderWeekly()
 */


// ════════════════════════════════════════════════════════════════════════════
// HTML TO ADD IN view-daily (in app.js, inside <section id="view-daily">)
// Add AFTER the cross-asset section (◆ 06), BEFORE report-actions:
// ════════════════════════════════════════════════════════════════════════════

const DAILY_HTML_ADDITIONS = `

  <!-- ◆ 07 – Cross-Asset Signals -->
  <div class="block" id="signalsBannerBlock" style="display:none">
    <div class="sec-header">
      <span class="sec-num">◆ 07</span>
      <span class="sec-title" data-i18n="signals">Cross-Asset Signals</span>
      <span class="sec-spacer"></span>
      <span class="sec-note signal-badge">⚡ Live correlation engine</span>
    </div>
    <div class="signals-list" id="signalsList"></div>
  </div>

  <!-- ◆ 08 – GCC Intelligence -->
  <div class="block" id="gccBlock" style="display:none">
    <div class="sec-header">
      <span class="sec-num">◆ 08</span>
      <span class="sec-title" data-i18n="gccIntel">GCC Intelligence</span>
      <span class="sec-spacer"></span>
      <span class="sec-note">Regional · MENA</span>
    </div>
    <div class="gcc-grid" id="gccGrid"></div>
    <div class="gcc-headlines" id="gccHeadlines"></div>
  </div>

  <!-- ◆ News Attribution -->
  <div class="news-attr" id="newsAttribution" style="display:none">
    <span class="news-attr-label">Intelligence sourced from</span>
    <span class="news-attr-sources" id="newsSourcesList"></span>
    <span class="news-attr-time" id="newsAttrTime"></span>
  </div>
`;


// ════════════════════════════════════════════════════════════════════════════
// CSS TO ADD inside <style> in app.js HTML
// ════════════════════════════════════════════════════════════════════════════

const NEW_CSS = `
  /* ── Cross-Asset Signals ── */
  .signal-badge{color:var(--gold);font-family:var(--font-mono);font-size:10px;letter-spacing:.1em;}
  .signals-list{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);}
  .signal-item{
    background:var(--bg-1);padding:14px 22px;
    display:grid;grid-template-columns:22px 1fr;gap:14px;align-items:start;
  }
  .signal-icon{color:var(--gold);font-size:14px;padding-top:1px;}
  .signal-text{font-family:var(--font-ui);font-size:13.5px;color:var(--fg-70);line-height:1.5;}
  .signal-text strong{color:var(--fg);font-weight:600;}

  /* ── GCC Grid ── */
  .gcc-grid{
    display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
    background:var(--line);border:1px solid var(--line);margin-bottom:1px;
  }
  .gcc-cell{background:var(--bg-1);padding:18px 22px;display:flex;flex-direction:column;gap:8px;}
  .gcc-label{font-family:var(--font-mono);font-size:10px;letter-spacing:.2em;color:var(--fg-50);text-transform:uppercase;}
  .gcc-value{font-family:var(--font-display);font-size:18px;font-weight:500;color:var(--fg);line-height:1.3;}
  .gcc-note{font-family:var(--font-ui);font-size:12.5px;color:var(--fg-70);line-height:1.5;}

  /* ── GCC Headlines ── */
  .gcc-headlines{
    border:1px solid var(--line);border-top:none;background:var(--bg-1);
    padding:16px 22px;display:grid;gap:10px;
  }
  .gcc-hl-head{font-family:var(--font-mono);font-size:10px;letter-spacing:.2em;color:var(--fg-50);text-transform:uppercase;margin-bottom:4px;}
  .gcc-hl-item{
    display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:start;
    font-family:var(--font-ui);font-size:13px;color:var(--fg);line-height:1.4;
  }
  .gcc-hl-source{
    font-family:var(--font-mono);font-size:10px;color:var(--gold);
    padding:2px 6px;border:1px solid rgba(184,148,74,0.35);white-space:nowrap;
    height:fit-content;margin-top:1px;
  }

  /* ── News Attribution ── */
  .news-attr{
    display:flex;flex-wrap:wrap;align-items:center;gap:10px;
    padding:14px 22px;border:1px solid var(--line);border-top:none;
    font-family:var(--font-mono);font-size:10px;
    background:var(--bg-0);margin-bottom:16px;
  }
  .news-attr-label{color:var(--fg-50);letter-spacing:.1em;text-transform:uppercase;}
  .news-attr-sources{color:var(--fg-70);letter-spacing:.04em;}
  .news-attr-time{color:var(--fg-30);margin-inline-start:auto;}

  @media(max-width:960px){.gcc-grid{grid-template-columns:1fr 1fr;}}
  @media(max-width:640px){.gcc-grid{grid-template-columns:1fr;}}
`;


// ════════════════════════════════════════════════════════════════════════════
// JS FUNCTIONS — Add these to the <script> block in app.js
// ════════════════════════════════════════════════════════════════════════════

/**
 * Render Cross-Asset Signals banner from the signals string.
 * Parses the ⚡ SIGNAL_TYPE: description format produced by the Python engine.
 */
function renderSignals(signalsText) {
  const block  = document.getElementById('signalsBannerBlock');
  const listEl = document.getElementById('signalsList');
  if (!block || !listEl || !signalsText) return;

  // Parse lines starting with ⚡
  const lines = String(signalsText).split('\n')
    .map(l => l.trim())
    .filter(l => l.startsWith('⚡'));

  if (!lines.length) return;

  block.style.display = '';
  listEl.innerHTML = lines.map(line => {
    // Bold the SIGNAL_TYPE label before the colon
    const formatted = line
      .replace(/^⚡\s*/, '')
      .replace(/^([A-Z][A-Z\- /]+):\s*/, '<strong>$1</strong> — ');
    return `<div class="signal-item">
      <div class="signal-icon">⚡</div>
      <div class="signal-text">${formatted}</div>
    </div>`;
  }).join('');
}

/**
 * Render GCC Intelligence section.
 * Expects d.gcc_intelligence from the enriched daily update.
 */
function renderGccIntelligence(d) {
  const block    = document.getElementById('gccBlock');
  const gridEl   = document.getElementById('gccGrid');
  const hlEl     = document.getElementById('gccHeadlines');
  if (!block || !gridEl || !hlEl) return;

  const gcc = d.gcc_intelligence;
  if (!gcc) return;

  block.style.display = '';

  // Determine direction classes
  function dirCls(val) {
    if (typeof val !== 'number') return 'fl';
    return val > 0.2 ? 'up' : val < -0.2 ? 'dn' : 'fl';
  }
  function fmtPctLocal(val) {
    if (typeof val !== 'number') return '—';
    return `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
  }

  const tasi   = gcc.tasi   || {};
  const oilLnk = gcc.oil_link || {};
  const clsT   = dirCls(tasi.d1_pct);
  const clsO   = dirCls(oilLnk.uso_d1);
  const arrowT = clsT === 'up' ? '▲' : clsT === 'dn' ? '▼' : '·';
  const arrowO = clsO === 'up' ? '▲' : clsO === 'dn' ? '▼' : '·';

  gridEl.innerHTML = `
    <div class="gcc-cell">
      <div class="gcc-label">Saudi TASI</div>
      <div class="gcc-value xa-change ${clsT}">${arrowT} ${fmtPctLocal(tasi.d1_pct)}</div>
      <div class="gcc-note">${esc(tasi.note || '')}</div>
    </div>
    <div class="gcc-cell">
      <div class="gcc-label">Oil (WTI) · GCC Link</div>
      <div class="gcc-value xa-change ${clsO}">${arrowO} ${fmtPctLocal(oilLnk.uso_d1)}</div>
      <div class="gcc-note">${esc(oilLnk.gcc_impact || '')}</div>
    </div>
    <div class="gcc-cell">
      <div class="gcc-label">Dollar Context</div>
      <div class="gcc-value" style="font-family:var(--font-ui);font-size:14px;color:var(--fg-70)">
        ${esc(gcc.dollar_context || '—')}
      </div>
    </div>`;

  // GCC Headlines
  const headlines = gcc.gcc_headlines || [];
  if (headlines.length) {
    hlEl.innerHTML = `
      <div class="gcc-hl-head">MENA Headlines · Live</div>
      ${headlines.map(h => {
        // Parse "[Source] Title" format
        const match = h.match(/^•?\s*\[([^\]]+)\]\s*(.+)$/);
        const source = match ? match[1] : '';
        const title  = match ? match[2] : h;
        return `<div class="gcc-hl-item">
          <span class="gcc-hl-source">${esc(source)}</span>
          <span>${esc(title)}</span>
        </div>`;
      }).join('')}`;
  } else {
    hlEl.style.display = 'none';
  }
}

/**
 * Render news attribution strip.
 */
function renderNewsAttribution(d) {
  const el      = document.getElementById('newsAttribution');
  const srcEl   = document.getElementById('newsSourcesList');
  const timeEl  = document.getElementById('newsAttrTime');
  if (!el || !srcEl || !timeEl) return;

  const sources    = d.news_sources || [];
  const fetchedAt  = d.news_fetched_at || '';

  if (!sources.length) return;
  el.style.display = '';
  srcEl.textContent = sources.join(' · ');
  timeEl.textContent = fetchedAt ? `Updated ${fetchedAt}` : '';
}

/**
 * Main upgrade call — add this to the END of renderDaily(d):
 *
 *   // ── Upgrade: signals, GCC intelligence, news attribution ──
 *   if (d.cross_asset_signals) renderSignals(d.cross_asset_signals);
 *   if (d.gcc_intelligence)    renderGccIntelligence(d);
 *   if (d.news_sources)        renderNewsAttribution(d);
 *
 * And update I18N object to add:
 *   en: { signals: 'Cross-Asset Signals', gccIntel: 'GCC Intelligence', ... }
 *   ar: { signals: 'إشارات الأصول المتقاطعة', gccIntel: 'استخبارات الخليج', ... }
 */


// ════════════════════════════════════════════════════════════════════════════
// FULL REPORT UPGRADE — replace renderReportBody with news-aware version
// ════════════════════════════════════════════════════════════════════════════

/**
 * Enhanced full report renderer — shows news attribution and GCC section
 * inside the report modal.
 * Replace the window.openFullReport function with this version.
 */
window.openFullReport = function(type) {
  const item = getItem(type);
  const wrap = type === 'weekly' ? (typeof _weeklyWrap !== 'undefined' ? _weeklyWrap : null) : (typeof _dailyWrap !== 'undefined' ? _dailyWrap : null);
  const raw  = (typeof CUR_LANG !== 'undefined' && CUR_LANG === 'ar'
    ? (wrap?.ar_full_report || item?._ar_full_report || item?._full_report)
    : item?._full_report) || '';
  if (!raw) return;

  const modal = document.getElementById('reportModal');
  document.getElementById('reportKicker').textContent =
    type === 'weekly' ? 'Client Weekly Brief' : 'Client Daily Brief';
  document.getElementById('reportTitle').textContent =
    type === 'weekly' ? 'EisaX Weekly Strategy Brief' : 'EisaX Daily Market Pulse';

  // Meta chips
  const bits = [];
  if (item?.date)              bits.push(`<span>${esc(item.date)}</span>`);
  if (item?.week_range)        bits.push(`<span>${esc(item.week_range)}</span>`);
  if (item?.market_regime)     bits.push(`<span>${esc(item.market_regime)}</span>`);
  if (item?.regime_confidence) bits.push(`<span>${esc(item.regime_confidence)} confidence</span>`);
  if (item?.data_timestamp)    bits.push(`<span>Data ${esc(fmtDateTime(item.data_timestamp))}</span>`);

  // News attribution chip
  const sources = item?.news_sources || [];
  if (sources.length) {
    bits.push(`<span style="color:var(--gold);border-color:rgba(184,148,74,0.4)">⚡ ${sources.length} live sources</span>`);
  }

  document.getElementById('reportMeta').innerHTML = bits.join('');

  // Report body: standard content + GCC appendix
  let bodyHTML = renderReportBody(raw);

  // Append GCC section if available
  if (type === 'daily' && item?.gcc_intelligence) {
    const gcc = item.gcc_intelligence;
    const headlines = (gcc.gcc_headlines || []).map(h => {
      const match = h.match(/^•?\s*\[([^\]]+)\]\s*(.+)$/);
      return match ? `<li><strong>${esc(match[1])}</strong> — ${esc(match[2])}</li>` : `<li>${esc(h)}</li>`;
    }).join('');

    bodyHTML += `
      <hr>
      <h2>GCC / MENA Intelligence</h2>
      <p>${esc(gcc.summary || '')}</p>
      <h3>Oil → GCC Link</h3>
      <p>${esc(gcc.oil_link?.gcc_impact || '')}</p>
      <h3>Dollar Context</h3>
      <p>${esc(gcc.dollar_context || '')}</p>
      ${headlines ? `<h3>MENA Headlines</h3><ul>${headlines}</ul>` : ''}`;
  }

  // Append cross-asset signals if available
  if (item?.cross_asset_signals) {
    const signals = String(item.cross_asset_signals).split('\n')
      .filter(l => l.trim().startsWith('⚡'))
      .map(l => `<li>${esc(l.replace(/^⚡\s*/, ''))}</li>`)
      .join('');
    if (signals) {
      bodyHTML += `<hr><h2>Cross-Asset Signals</h2><ul>${signals}</ul>`;
    }
  }

  // News attribution
  if (sources.length) {
    bodyHTML += `
      <hr>
      <p style="font-family:var(--font-mono);font-size:11px;color:var(--fg-50)">
        Intelligence sourced from: ${sources.map(s => `<strong>${esc(s)}</strong>`).join(', ')}
        ${item.news_fetched_at ? `· fetched ${esc(item.news_fetched_at)}` : ''}
      </p>`;
  }

  document.getElementById('reportBody').innerHTML = bodyHTML;
  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
};


// ════════════════════════════════════════════════════════════════════════════
// I18N ADDITIONS — merge into the existing I18N object
// ════════════════════════════════════════════════════════════════════════════

const I18N_ADDITIONS = {
  en: {
    signals:     'Cross-Asset Signals',
    gccIntel:    'GCC Intelligence',
    menaHeadlines: 'MENA Headlines',
    liveEngine:  'Live correlation engine',
    sourcedFrom: 'Intelligence sourced from',
  },
  ar: {
    signals:     'إشارات الأصول المتقاطعة',
    gccIntel:    'استخبارات الخليج',
    menaHeadlines: 'عناوين منطقة الشرق الأوسط',
    liveEngine:  'محرك الارتباط المباشر',
    sourcedFrom: 'الاستخبارات مصدرها',
  },
};

// To use: merge I18N_ADDITIONS into your existing I18N object:
// Object.assign(I18N.en, I18N_ADDITIONS.en);
// Object.assign(I18N.ar, I18N_ADDITIONS.ar);
