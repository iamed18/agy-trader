(function () {
  "use strict";

  const REFRESH_MS = 30000;
  let activeTab = "model_news";
  let lastIntelPayload = null;
  let refreshTimer = null;

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`${url}: HTTP ${resp.status}`);
    }
    return resp.json();
  }

  function fmtMoney(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
    return "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtPct(v, signed = true) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
    const n = Number(v);
    const prefix = signed && n > 0 ? "+" : "";
    return prefix + n.toFixed(2) + "%";
  }

  function clsSign(v) {
    if (v === null || v === undefined) return "";
    return Number(v) >= 0 ? "pos" : "neg";
  }

  function renderLineChart(container, series, lines, yUnit = "") {
    if (!series || series.length === 0) {
      container.innerHTML = '<div class="muted">No portfolio snapshots recorded yet.</div>';
      return;
    }

    const width = 680;
    const height = 220;
    const padL = 48;
    const padR = 16;
    const padT = 16;
    const padB = 26;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const allVals = [];
    for (const pt of series) {
      for (const ln of lines) {
        const val = pt[ln.key];
        if (val !== null && val !== undefined && Number.isFinite(Number(val))) {
          allVals.push(Number(val));
        }
      }
    }
    if (allVals.length === 0) {
      container.innerHTML = '<div class="muted">Waiting for series values…</div>';
      return;
    }

    let minY = Math.min(...allVals);
    let maxY = Math.max(...allVals);
    if (Math.abs(maxY - minY) < 1e-6) {
      minY -= 1;
      maxY += 1;
    } else {
      const margin = (maxY - minY) * 0.1;
      minY -= margin;
      maxY += margin;
    }

    const xFor = (i) =>
      series.length === 1 ? padL + plotW / 2 : padL + (i / (series.length - 1)) * plotW;
    const yFor = (v) => padT + (1 - (v - minY) / (maxY - minY)) * plotH;

    let gridSvg = "";
    for (let g = 0; g <= 4; g++) {
      const yVal = minY + ((maxY - minY) * g) / 4;
      const yPx = yFor(yVal);
      gridSvg += `<line x1="${padL}" y1="${yPx.toFixed(1)}" x2="${width - padR}" y2="${yPx.toFixed(1)}" stroke="#232e42" stroke-dasharray="3,3" />`;
      gridSvg += `<text x="${padL - 6}" y="${(yPx + 4).toFixed(1)}" text-anchor="end" fill="#8b9bb4" font-size="10">${yVal.toFixed(1)}${esc(yUnit)}</text>`;
    }

    let pathsSvg = "";
    for (const ln of lines) {
      const coords = [];
      series.forEach((pt, idx) => {
        const v = pt[ln.key];
        if (v !== null && v !== undefined && Number.isFinite(Number(v))) {
          coords.push(`${xFor(idx).toFixed(1)},${yFor(Number(v)).toFixed(1)}`);
        }
      });
      if (coords.length === 1) {
        const [cx, cy] = coords[0].split(",");
        pathsSvg += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="${ln.color}" />`;
      } else if (coords.length > 1) {
        pathsSvg += `<polyline fill="none" stroke="${ln.color}" stroke-width="2" points="${coords.join(" ")}" />`;
      }
    }

    const firstTs = String(series[0].timestamp || "").slice(0, 10);
    const lastTs = String(series[series.length - 1].timestamp || "").slice(0, 16).replace("T", " ");
    const xLabels = `
      <text x="${padL}" y="${height - 6}" fill="#8b9bb4" font-size="10">${esc(firstTs)}</text>
      <text x="${width - padR}" y="${height - 6}" text-anchor="end" fill="#8b9bb4" font-size="10">${esc(lastTs)}</text>
    `;

    container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" preserveAspectRatio="none">${gridSvg}${pathsSvg}${xLabels}</svg>`;
  }

  function renderOverview(data) {
    document.getElementById("kpi-equity").textContent = fmtMoney(data.equity);
    document.getElementById("kpi-cash").textContent = `Cash: ${fmtMoney(data.cash)} · BP: ${fmtMoney(data.buying_power)}`;

    const totEl = document.getElementById("kpi-total-ret");
    totEl.textContent = fmtPct(data.total_return_pct);
    totEl.className = `kpi-value ${clsSign(data.total_return_pct)}`;
    document.getElementById("kpi-baselines").textContent = `SPY: ${fmtPct(data.spy_return_pct)} · EW-50: ${fmtPct(data.ew50_return_pct)}`;

    const dayEl = document.getElementById("kpi-daily-pnl");
    dayEl.textContent = fmtPct(data.daily_pnl_pct);
    dayEl.className = `kpi-value ${clsSign(data.daily_pnl_pct)}`;
    document.getElementById("kpi-drawdown").textContent = `DD: ${fmtPct(data.drawdown_pct, false)} · HWM: ${fmtMoney(data.high_water_mark)}`;

    document.getElementById("kpi-exposure").textContent =
      `Gross ${(Number(data.gross_exposure || 0) * 100).toFixed(1)}% · Net ${(Number(data.net_exposure || 0) * 100).toFixed(1)}%`;

    const dials = data.control_dials || {};
    const dialsPill = document.getElementById("dials-pill");
    if (dials.pause_new_entries) {
      dialsPill.textContent = "Dials: PAUSED";
      dialsPill.className = "badge badge-warn";
    } else {
      dialsPill.textContent = `Dials: ACTIVE (${Number(dials.exposure_multiplier ?? 1).toFixed(2)}x)`;
      dialsPill.className = "badge badge-ok";
    }
    document.getElementById("kpi-dials-detail").textContent =
      `Multiplier: ${Number(dials.exposure_multiplier ?? 1).toFixed(2)}x · Shorting: ${dials.allow_shorting ? "ON" : "OFF"}`;

    const banner = document.getElementById("blackout-banner");
    const bo = data.blackout || {};
    if (banner) {
      if (bo.active) {
        const isHalt = Boolean(bo.halted);
        banner.className = `blackout-banner ${isHalt ? "blackout-halt" : "blackout-freeze"}`;
        const frozenStr = (bo.frozen_symbols || []).join(", ") || "NONE";
        const etfSummary = (bo.etf_weights_summary || []).join(" · ");
        const detail = isHalt
          ? `ALL PORTFOLIO ORDERS HALTED (fail-closed): ${bo.halt_reason || "invalid BLACKOUT_PERIOD"}`
          : `Frozen symbols (zero buys/sells/cancels): ${frozenStr}${etfSummary ? ` · ETFs: ${etfSummary}` : ""}`;
        banner.innerHTML = `<strong>${isHalt ? "COMPLIANCE HALT" : "BLACKOUT_PERIOD ACTIVE"}:</strong> ${esc(detail)}`;
      } else {
        banner.className = "blackout-banner hidden";
        banner.innerHTML = "";
      }
    }
  }

  function renderPositions(data) {
    const rows = data.positions || [];
    document.getElementById("positions-count").textContent = `${rows.length} open`;
    const body = document.getElementById("positions-body");
    if (rows.length === 0) {
      body.innerHTML = '<tr><td colspan="9" class="empty">No open positions.</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((p) => {
        const syncBadge = p.stop_synced
          ? '<span class="badge badge-ok">GTC OK</span>'
          : '<span class="badge badge-err">MISSING</span>';
        const dist = p.distance_to_stop_pct !== null ? `${Number(p.distance_to_stop_pct).toFixed(2)}%` : "—";
        return `<tr>
          <td><strong>${esc(p.symbol)}</strong></td>
          <td>${esc(p.side)}</td>
          <td class="num">${Number(p.qty).toFixed(4).replace(/\.?0+$/, "")}</td>
          <td class="num">${fmtMoney(p.avg_entry_price)}</td>
          <td class="num">${fmtMoney(p.market_price)}</td>
          <td class="num ${clsSign(p.unrealized_pnl)}">${fmtMoney(p.unrealized_pnl)}</td>
          <td class="num">${p.stop_price !== null ? fmtMoney(p.stop_price) : "—"}</td>
          <td class="num">${dist}</td>
          <td>${syncBadge}</td>
        </tr>`;
      })
      .join("");
  }

  function renderOrders(data) {
    const rows = data.orders || [];
    document.getElementById("orders-count").textContent = `${rows.length} recent`;
    const body = document.getElementById("orders-body");
    if (rows.length === 0) {
      body.innerHTML = '<tr><td colspan="8" class="empty">No live orders recorded yet.</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((o) => {
        const slip =
          o.realized_slippage_bps !== null && o.realized_slippage_bps !== undefined
            ? `${Number(o.realized_slippage_bps).toFixed(1)} / ${Number(o.backtest_budget_bps).toFixed(0)}`
            : "—";
        const ts = String(o.submitted_at || "").slice(0, 19).replace("T", " ");
        return `<tr>
          <td class="muted">${esc(ts)}</td>
          <td><strong>${esc(o.symbol)}</strong></td>
          <td>${esc(o.side)}</td>
          <td>${esc(o.order_type)}</td>
          <td class="num">${Number(o.qty).toFixed(4).replace(/\.?0+$/, "")}</td>
          <td class="num">${o.filled_avg_price !== null ? fmtMoney(o.filled_avg_price) : "—"}</td>
          <td class="num">${slip}</td>
          <td><span class="badge">${esc(o.status)}</span></td>
        </tr>`;
      })
      .join("");
  }

  function renderHealth(data) {
    const walPill = document.getElementById("wal-pill");
    if (data.wal_ok) {
      walPill.textContent = "WAL: OK";
      walPill.className = "badge badge-ok";
    } else {
      walPill.textContent = `WAL: ${data.wal_detail || "ERROR"}`;
      walPill.className = "badge badge-err";
    }

    const counts = data.data_counts || {};
    document.getElementById("store-counts").textContent =
      `Parquet: ${counts.parquet_symbols ?? 0} syms · News: ${counts.news_articles ?? 0} (unscored ${counts.unscored_news ?? 0}) · SEC: ${counts.sec_filings ?? 0} · Macro: ${counts.macro_points ?? 0}`;

    const grid = document.getElementById("daemon-grid");
    const daemons = data.daemons || {};
    grid.innerHTML = Object.values(daemons)
      .map((d) => {
        const st = String(d.status || "missing");
        let cls = "badge-warn";
        if (st === "healthy" || st === "idle_market_closed") cls = "badge-ok";
        else if (st === "missing" || st === "error" || st === "critical") cls = "badge-err";
        const age = d.staleness_seconds !== null ? `${d.staleness_seconds}s ago` : "never";
        const rss = d.rss_mb !== null && d.rss_mb !== undefined ? `${d.rss_mb} MB` : "—";
        return `<div class="daemon-tile">
          <div class="daemon-top">
            <span>${esc(d.daemon_name)}</span>
            <span class="badge ${cls}">${esc(st)}</span>
          </div>
          <div class="daemon-meta">Heartbeat: ${esc(age)}</div>
          <div class="daemon-meta">RSS: ${esc(rss)}</div>
        </div>`;
      })
      .join("");
  }

  function renderIntelligence(data) {
    lastIntelPayload = data;
    const panel = document.getElementById("intel-panel");
    const items = (data && data[activeTab]) || [];
    if (items.length === 0) {
      panel.innerHTML = '<div class="muted">No items in this category yet.</div>';
      return;
    }

    if (activeTab === "sec_filings") {
      panel.innerHTML = items
        .map((ev) => {
          const ts = String(ev.accepted_at || "").slice(0, 19).replace("T", " ");
          const detail = ev.form_type === "4"
            ? `Net insider buy: ${fmtMoney(ev.net_insider_buy_usd)} (${ev.insider_buyer_count}B/${ev.insider_seller_count}S)`
            : `Items: ${(ev.items || []).join(", ") || "none"}`;
          return `<div class="intel-row">
            <div>
              <div class="intel-title">Form ${esc(ev.form_type)} · ${esc(ev.symbol)}</div>
              <div class="intel-sub">${esc(detail)} · Accession ${esc(ev.accession_number)}</div>
            </div>
            <span class="muted">${esc(ts)}</span>
          </div>`;
        })
        .join("");
      return;
    }

    if (activeTab === "audit_log") {
      panel.innerHTML = items
        .map((row) => {
          const ts = String(row.changed_at || "").slice(0, 19).replace("T", " ");
          const d = row.dials || {};
          return `<div class="intel-row">
            <div>
              <div class="intel-title">${esc(row.actor)}: ${esc(row.reason)}</div>
              <div class="intel-sub">pause=${esc(d.pause_new_entries)} · mult=${esc(d.exposure_multiplier)} · shorts=${esc(d.allow_shorting)}</div>
            </div>
            <span class="muted">${esc(ts)}</span>
          </div>`;
        })
        .join("");
      return;
    }

    panel.innerHTML = items
      .map((art) => {
        const ts = String(art.published_at || "").slice(0, 19).replace("T", " ");
        const score =
          art.sentiment_score !== null && art.sentiment_score !== undefined
            ? Number(art.sentiment_score).toFixed(2)
            : "unscored";
        const badgeCls =
          art.sentiment_label === "positive"
            ? "badge-ok"
            : art.sentiment_label === "negative"
              ? "badge-err"
              : "";
        return `<div class="intel-row">
          <div>
            <div class="intel-title">${esc(art.headline)}</div>
            <div class="intel-sub">${esc(art.source)} · [${esc((art.symbols || []).join(", "))}] · ${esc(ts)}</div>
          </div>
          <span class="badge ${badgeCls}">${esc(art.sentiment_label || "pending")} (${esc(score)})</span>
        </div>`;
      })
      .join("");
  }

  async function refreshAll() {
    try {
      const [overview, curve, positions, orders, health, intel] = await Promise.all([
        fetchJson("/api/overview"),
        fetchJson("/api/equity-curve?limit=1000"),
        fetchJson("/api/positions"),
        fetchJson("/api/orders?limit=100"),
        fetchJson("/api/health"),
        fetchJson("/api/intelligence?limit=50"),
      ]);

      renderOverview(overview);
      renderLineChart(
        document.getElementById("equity-chart"),
        curve.series || [],
        [
          { key: "portfolio_index", color: "#4ea1ff" },
          { key: "spy_index", color: "#fbbf24" },
          { key: "ew50_index", color: "#a78bfa" },
        ]
      );
      renderLineChart(
        document.getElementById("drawdown-chart"),
        curve.series || [],
        [{ key: "drawdown_pct", color: "#f87171" }],
        "%"
      );
      renderPositions(positions);
      renderOrders(orders);
      renderHealth(health);
      renderIntelligence(intel);
      document.getElementById("last-updated").textContent =
        "Last refresh: " + new Date().toLocaleTimeString();
    } catch (err) {
      document.getElementById("last-updated").textContent = "Refresh error: " + String(err);
    }
  }

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab || "model_news";
      if (lastIntelPayload) {
        renderIntelligence(lastIntelPayload);
      }
    });
  });

  document.getElementById("refresh-btn").addEventListener("click", refreshAll);

  refreshAll();
  refreshTimer = window.setInterval(refreshAll, REFRESH_MS);
  window.addEventListener("beforeunload", () => {
    if (refreshTimer !== null) {
      window.clearInterval(refreshTimer);
      refreshTimer = null;
    }
  });
})();
