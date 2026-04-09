'use strict';

const API = {
  liveData: '/api/live-data?seed=1001',
  summary: '/api/summary',
  openenv: '/openenv',
};

const STATE = {
  payload: null,
  step: 0,
  running: false,
  speed: 1,
  frameId: null,
  lastFrame: 0,
  msPerStep: 110,
  charts: {},
};

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(13,21,33,0.96)',
      titleColor: '#00f5d4',
      bodyColor: '#c9d3df',
      borderColor: 'rgba(0,245,212,0.18)',
      borderWidth: 1,
    },
  },
  scales: {
    x: {
      grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
      ticks: { color: '#7e8ca0', maxTicksLimit: 8, font: { size: 10 } },
    },
    y: {
      grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
      ticks: { color: '#7e8ca0', font: { size: 10 } },
    },
  },
};

function el(id) {
  return document.getElementById(id);
}

function money(v) {
  return `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function pct(v) {
  return `${Number(v).toFixed(2)}%`;
}

function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function buildPriceChart() {
  const ctx = el('priceChart');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Price',
          data: [],
          borderColor: '#00f5d4',
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          backgroundColor: (context) => {
            const g = context.chart.ctx.createLinearGradient(0, 0, 0, 220);
            g.addColorStop(0, 'rgba(0,245,212,0.16)');
            g.addColorStop(1, 'rgba(0,245,212,0.01)');
            return g;
          },
          order: 3,
        },
        {
          label: 'MA5',
          data: [],
          borderColor: 'rgba(245,166,35,0.65)',
          borderWidth: 1,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.25,
          fill: false,
          order: 2,
        },
        {
          label: 'MA10',
          data: [],
          borderColor: 'rgba(167,139,250,0.55)',
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0.25,
          fill: false,
          order: 1,
        },
        {
          label: 'BUY',
          data: [],
          type: 'scatter',
          pointStyle: 'triangle',
          rotation: 0,
          pointRadius: 6,
          backgroundColor: '#22c55e',
          borderColor: '#22c55e',
          order: 0,
        },
        {
          label: 'SELL',
          data: [],
          type: 'scatter',
          pointStyle: 'triangle',
          rotation: 180,
          pointRadius: 6,
          backgroundColor: '#ef4444',
          borderColor: '#ef4444',
          order: 0,
        },
      ],
    },
    options: {
      ...CHART_OPTS,
      plugins: { ...CHART_OPTS.plugins, legend: { display: false } },
      scales: {
        x: { ...CHART_OPTS.scales.x },
        y: { ...CHART_OPTS.scales.y, ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => `$${Number(v).toFixed(0)}` } },
      },
    },
  });
}

function buildPortfolioChart() {
  const ctx = el('portfolioChart');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Agent Portfolio',
          data: [],
          borderColor: '#a78bfa',
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          backgroundColor: (context) => {
            const g = context.chart.ctx.createLinearGradient(0, 0, 0, 160);
            g.addColorStop(0, 'rgba(167,139,250,0.16)');
            g.addColorStop(1, 'rgba(167,139,250,0.01)');
            return g;
          },
        },
        {
          label: 'Buy & Hold',
          data: [],
          borderColor: 'rgba(245,166,35,0.8)',
          borderWidth: 1.4,
          borderDash: [5, 3],
          pointRadius: 0,
          tension: 0.25,
          fill: false,
        },
        {
          label: 'Initial',
          data: [],
          borderColor: 'rgba(255,255,255,0.14)',
          borderWidth: 1,
          borderDash: [2, 4],
          pointRadius: 0,
          tension: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...CHART_OPTS,
      plugins: {
        ...CHART_OPTS.plugins,
        legend: { display: true, labels: { color: '#8b9ab3', boxWidth: 12, font: { size: 10 } } },
      },
      scales: {
        x: { ...CHART_OPTS.scales.x },
        y: { ...CHART_OPTS.scales.y, ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => `$${(Number(v) / 1000).toFixed(1)}k` } },
      },
    },
  });
}

function buildHeroChart() {
  const ctx = el('heroMiniChart');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      ...CHART_OPTS,
      plugins: { ...CHART_OPTS.plugins, legend: { display: false } },
      scales: {
        x: { display: false },
        y: { ...CHART_OPTS.scales.y, ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => `$${Number(v).toFixed(0)}` } },
      },
    },
  });
}

function buildEpsilonChart() {
  const ctx = el('epsilonChart');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: '#a78bfa', borderWidth: 1.4, pointRadius: 0, tension: 0.25, fill: true, backgroundColor: 'rgba(167,139,250,0.1)' }] },
    options: {
      ...CHART_OPTS,
      scales: {
        x: { ...CHART_OPTS.scales.x },
        y: { ...CHART_OPTS.scales.y, min: 0, max: 1.05, ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => Number(v).toFixed(2) } },
      },
    },
  });
}

function buildComparisonChart() {
  const ctx = el('comparisonChart');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: ['Live Market'], datasets: [] },
    options: {
      ...CHART_OPTS,
      plugins: {
        ...CHART_OPTS.plugins,
        legend: { display: true, labels: { color: '#8b9ab3', boxWidth: 12, font: { size: 11 } } },
      },
      scales: {
        x: { ...CHART_OPTS.scales.x },
        y: { ...CHART_OPTS.scales.y, ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => `${v}%` } },
      },
    },
  });
}

function setBadge(text, kind) {
  const badge = el('actionBadge');
  const icon = badge?.querySelector('.action-icon');
  const textEl = el('actionText');
  if (!badge || !icon || !textEl) return;
  badge.className = 'action-badge';
  if (kind) badge.classList.add(kind);
  textEl.textContent = text;
  icon.textContent = kind === 'act-buy' ? '▲' : kind === 'act-sell' ? '▼' : kind === 'act-hold' ? '■' : '⏸';
}

function setSourceBanner(source, ticker, period, interval) {
  const navBadge = document.querySelector('.nav-badge');
  if (!navBadge) return;
  navBadge.innerHTML = `<span class="badge-dot"></span>${source.toUpperCase()} ${ticker} ${period}/${interval}`;
}

function renderStaticConfig(summary) {
  const env = [
    ['num_steps', summary.length],
    ['initial_balance', '$10,000'],
    ['min_balance', '$1,000'],
    ['transaction_fee_pct', '0.1%'],
    ['max_shares', 100],
    ['stop_loss_pct', '10%'],
    ['state_size', summary.stateSize],
    ['action_size', 3],
  ];

  const dqn = [
    ['hidden_dim_1', 64],
    ['hidden_dim_2', 64],
    ['learning_rate', '1e-3'],
    ['gamma', 0.99],
    ['batch_size', 64],
    ['replay_buffer_size', '50,000'],
    ['epsilon_start', 1.0],
    ['epsilon_end', 0.01],
    ['epsilon_decay', 0.995],
    ['target_update_freq', '10 ep'],
    ['min_trade_advantage', 0.05],
  ];

  const reward = [
    ['profit_scale', 1.0],
    ['drawdown_scale', 0.10],
    ['trade_penalty', 0.004],
    ['repeat_trade_penalty', 0.002],
    ['fee_scale', 1.0],
    ['slippage_scale', 1.0],
    ['positive_step_bonus', 0.002],
    ['negative_step_penalty', 0.002],
  ];

  const renderRows = (id, rows) => {
    const target = el(id);
    if (!target) return;
    target.innerHTML = rows
      .map(([k, v]) => `<div class="config-row"><span class="config-key">${k}</span><span class="config-val">${v}</span></div>`)
      .join('');
  };

  renderRows('envConfig', env);
  renderRows('priceConfig', [['source', summary.source], ['ticker', summary.ticker], ['period', summary.period], ['interval', summary.interval]]);
  renderRows('dqnConfig', dqn);
  renderRows('rewardConfig', reward);
}

function computeBaseline(prices) {
  const initial = prices[0];
  const shares = Math.floor(10000 / initial);
  const cash = 10000 - shares * initial;
  return prices.map((p) => cash + shares * p);
}

function updateFrame(t) {
  const { history, prices, baseline, metrics, source } = STATE.payload;
  const total = history.price.length;
  const idx = Math.min(t, total - 1);
  const labels = Array.from({ length: idx + 1 }, (_, i) => i);

  const priceSlice = history.price.slice(0, idx + 1);
  const ma5Slice = history.ma5.slice(0, idx + 1);
  const ma10Slice = history.ma10.slice(0, idx + 1);
  const buyPoints = history.action.slice(0, idx + 1).map((a, i) => (a === 1 ? history.price[i] : null));
  const sellPoints = history.action.slice(0, idx + 1).map((a, i) => (a === 2 ? history.price[i] : null));
  const portSlice = history.portfolioValue.slice(0, idx + 1);
  const bhSlice = baseline.slice(0, idx + 1);
  const initSlice = Array(idx + 1).fill(10000);

  STATE.charts.price.data.labels = labels;
  STATE.charts.price.data.datasets[0].data = priceSlice;
  STATE.charts.price.data.datasets[1].data = ma5Slice;
  STATE.charts.price.data.datasets[2].data = ma10Slice;
  STATE.charts.price.data.datasets[3].data = buyPoints;
  STATE.charts.price.data.datasets[4].data = sellPoints;
  STATE.charts.price.update('none');

  STATE.charts.portfolio.data.labels = labels;
  STATE.charts.portfolio.data.datasets[0].data = portSlice;
  STATE.charts.portfolio.data.datasets[1].data = bhSlice;
  STATE.charts.portfolio.data.datasets[2].data = initSlice;
  STATE.charts.portfolio.update('none');

  STATE.charts.hero.data.labels = labels;
  STATE.charts.hero.data.datasets = [
    {
      label: 'Live Price',
      data: priceSlice,
      borderColor: '#00f5d4',
      borderWidth: 1.6,
      pointRadius: 0,
      fill: true,
      tension: 0.28,
      backgroundColor: (context) => {
        const g = context.chart.ctx.createLinearGradient(0, 0, 0, 280);
        g.addColorStop(0, 'rgba(0,245,212,0.16)');
        g.addColorStop(1, 'rgba(0,245,212,0.01)');
        return g;
      },
    },
    {
      label: 'BUY',
      type: 'scatter',
      data: buyPoints,
      pointStyle: 'triangle',
      rotation: 0,
      pointRadius: 5,
      backgroundColor: '#22c55e',
      borderColor: '#22c55e',
    },
    {
      label: 'SELL',
      type: 'scatter',
      data: sellPoints,
      pointStyle: 'triangle',
      rotation: 180,
      pointRadius: 5,
      backgroundColor: '#ef4444',
      borderColor: '#ef4444',
    },
  ];
  STATE.charts.hero.update('none');

  const pv = history.portfolioValue[idx];
  const bal = history.balance[idx];
  const shares = history.sharesHeld[idx];
  const profit = pv - 10000;
  const returnPct = (profit / 10000) * 100;
  const totalTrades = history.action.slice(0, idx + 1).filter((a) => a !== 0).length;
  const winRate = idx > 0 ? history.portfolioValue.slice(1, idx + 1).filter((v, i) => v > history.portfolioValue[i]).length / idx : 0;
  let peak = 10000;
  let maxDd = 0;
  for (let i = 0; i <= idx; i++) {
    peak = Math.max(peak, history.portfolioValue[i]);
    maxDd = Math.max(maxDd, (peak - history.portfolioValue[i]) / peak);
  }

  el('metPortfolio').textContent = money(pv);
  const change = el('metPortfolioChange');
  change.textContent = `${profit >= 0 ? '+' : ''}${money(profit)} (${returnPct >= 0 ? '+' : ''}${pct(returnPct)})`;
  change.className = `metric-change ${profit >= 0 ? 'positive' : 'negative'}`;
  el('metBalance').textContent = money(bal);
  el('metShares').textContent = String(Math.round(shares));
  el('metWinRate').textContent = idx > 0 ? pct(winRate * 100) : '—';
  el('metDrawdown').textContent = pct(maxDd * 100);
  el('metTrades').textContent = String(totalTrades);
  setBadge(history.action[idx] === 1 ? 'BUY' : history.action[idx] === 2 ? 'SELL' : 'HOLD', history.action[idx] === 1 ? 'act-buy' : history.action[idx] === 2 ? 'act-sell' : 'act-hold');
  el('stepCounter').textContent = `${idx + 1} / ${total}`;

  const episodes = [1, 2, 3, 4, 5];
  const liveReturn = metrics.return_pct;
  const bhReturn = ((baseline[baseline.length - 1] - 10000) / 10000) * 100;
  STATE.charts.compare.data.labels = ['Live Market', 'Buy & Hold'];
  STATE.charts.compare.data.datasets = [
    {
      label: 'Return %',
      data: [liveReturn, bhReturn],
      backgroundColor: [liveReturn >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)', 'rgba(245,166,35,0.45)'],
      borderRadius: 6,
    },
  ];
  STATE.charts.compare.update('none');

  const tbody = el('perfTableBody');
  if (tbody) {
    tbody.innerHTML = `
      <tr>
        <td style="color:#8b9ab3">Live</td>
        <td class="${profit >= 0 ? 'td-pos' : 'td-neg'}">${profit >= 0 ? '+' : ''}${money(profit)}</td>
        <td class="${returnPct >= 0 ? 'td-pos' : 'td-neg'}">${returnPct >= 0 ? '+' : ''}${pct(returnPct)}</td>
        <td class="td-neutral">${(winRate * 100).toFixed(1)}%</td>
        <td class="td-neg">−${pct(maxDd * 100)}</td>
        <td style="color:#8b9ab3">${totalTrades}</td>
        <td class="td-neutral">${source.toUpperCase()}</td>
      </tr>
    `;
  }
}

function stopPlayback() {
  STATE.running = false;
  if (STATE.frameId) cancelAnimationFrame(STATE.frameId);
  const play = el('btnPlay');
  const pause = el('btnPause');
  if (play) play.disabled = false;
  if (pause) pause.disabled = true;
}

function playPlayback() {
  if (!STATE.payload) return;
  STATE.running = true;
  const play = el('btnPlay');
  const pause = el('btnPause');
  if (play) play.disabled = true;
  if (pause) pause.disabled = false;
  let last = 0;
  const tick = (ts) => {
    if (!STATE.running) return;
    if (ts - last < STATE.msPerStep / STATE.speed) {
      STATE.frameId = requestAnimationFrame(tick);
      return;
    }
    last = ts;
    if (STATE.step >= STATE.payload.history.price.length) {
      stopPlayback();
      return;
    }
    updateFrame(STATE.step);
    STATE.step += 1;
    STATE.frameId = requestAnimationFrame(tick);
  };
  STATE.frameId = requestAnimationFrame(tick);
}

function resetPlayback() {
  stopPlayback();
  STATE.step = 0;
  if (STATE.payload) updateFrame(0);
}

async function loadLiveData() {
  const res = await fetch(API.liveData, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch live data: ${res.status}`);
  }
  const payload = await res.json();
  payload.baseline = computeBaseline(payload.prices);
  STATE.payload = payload;
  return payload;
}

function animateCounters() {
  document.querySelectorAll('.stat-num').forEach((el) => {
    const target = parseInt(el.dataset.target, 10);
    let current = 0;
    const timer = setInterval(() => {
      current = Math.min(target, current + Math.max(1, Math.ceil(target / 40)));
      el.textContent = String(current);
      if (current >= target) clearInterval(timer);
    }, 24);
  });
}

function initParticles() {
  const container = el('heroParticles');
  if (!container) return;
  for (let i = 0; i < 24; i += 1) {
    const dot = document.createElement('div');
    dot.className = 'particle';
    dot.style.left = `${Math.random() * 100}%`;
    dot.style.top = `${Math.random() * 100}%`;
    dot.style.animationDuration = `${6 + Math.random() * 8}s`;
    dot.style.animationDelay = `${Math.random() * 8}s`;
    dot.style.width = `${1 + Math.random() * 2.5}px`;
    dot.style.height = dot.style.width;
    container.appendChild(dot);
  }
}

function initNavbar() {
  const nav = el('navbar');
  window.addEventListener('scroll', () => {
    nav?.classList.toggle('scrolled', window.scrollY > 40);
  }, { passive: true });
}

function initEpsilonChart() {
  const chart = STATE.charts.epsilon;
  if (!chart) return;
  const eps = [];
  let e = 1.0;
  for (let i = 0; i < 500; i += 1) {
    eps.push(e);
    e = Math.max(0.01, e * 0.995);
  }
  chart.data.labels = eps.map((_, i) => i);
  chart.data.datasets[0].data = eps;
  chart.update('none');
}

function renderSummary(payload) {
  const summary = {
    length: payload.history.price.length,
    stateSize: 7,
    source: payload.source,
    ticker: payload.ticker,
    period: payload.period,
    interval: payload.interval,
  };
  renderStaticConfig(summary);

  const kpiReturn = el('kpiAvgReturn');
  if (kpiReturn) kpiReturn.textContent = `${payload.metrics.return_pct >= 0 ? '+' : ''}${payload.metrics.return_pct.toFixed(1)}%`;

  const kpiBh = el('kpiBHReturn');
  if (kpiBh) {
    const bh = ((payload.baseline[payload.baseline.length - 1] - 10000) / 10000) * 100;
    kpiBh.textContent = `${bh >= 0 ? '+' : ''}${bh.toFixed(1)}%`;
  }

  const kpiWin = el('kpiWin');
  if (kpiWin) kpiWin.textContent = `${(payload.metrics.win_rate * 100).toFixed(1)}%`;

  const kpiDD = el('kpiDD');
  if (kpiDD) kpiDD.textContent = `−${(payload.metrics.max_drawdown * 100).toFixed(1)}%`;
}

async function loadOpenEnvDiscovery() {
  const res = await fetch(API.openenv, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch OpenEnv discovery: ${res.status}`);
  }
  return res.json();
}

function renderOpenEnvSpotlight(payload) {
  const pathEl = el('openenvPath');
  const descEl = el('openenvDescription');
  const methodsEl = el('openenvMethods');
  const exampleEl = el('openenvExample');

  if (pathEl) pathEl.textContent = `GET ${payload.path}`;
  if (descEl) descEl.textContent = payload.description;
  if (methodsEl && Array.isArray(payload.methods)) {
    methodsEl.innerHTML = payload.methods
      .map((method) => `<span class="hero-api-method ${method.toLowerCase()}">${method}</span>`)
      .join('');
  }
  if (exampleEl && Array.isArray(payload.post_examples) && payload.post_examples.length) {
    exampleEl.textContent = JSON.stringify(payload.post_examples[payload.post_examples.length - 1], null, 2);
  }
}

async function boot() {
  initNavbar();
  initParticles();
  animateCounters();

  document.title = 'StockRL - Live Market Dashboard';
  const navLinks = document.querySelectorAll('.nav-link');
  if (navLinks[0]) navLinks[0].textContent = 'Live Market';
  const heroSub = document.querySelector('.hero-title-sub');
  if (heroSub) heroSub.textContent = 'Trading Dashboard';
  const heroDesc = document.querySelector('.hero-desc');
  if (heroDesc) {
    heroDesc.innerHTML = 'A production-quality DQN agent built with <strong>PyTorch</strong> that renders <em>live market bars</em>, portfolio value, and model actions against real price data.';
  }

  STATE.charts.hero = buildHeroChart();
  STATE.charts.price = buildPriceChart();
  STATE.charts.portfolio = buildPortfolioChart();
  STATE.charts.epsilon = buildEpsilonChart();
  STATE.charts.compare = buildComparisonChart();

  try {
    const discovery = await loadOpenEnvDiscovery();
    renderOpenEnvSpotlight(discovery);

    const payload = await loadLiveData();
    setSourceBanner(payload.source, payload.ticker, payload.period, payload.interval);
    renderSummary(payload);
    initEpsilonChart();
    updateFrame(0);
    const startBtn = el('startSimBtn');
    if (startBtn) {
      startBtn.querySelector('span').textContent = '▶ Play Live Feed';
    }
    const sectionTitle = document.querySelector('#live-market .section-title');
    if (sectionTitle) sectionTitle.textContent = 'Live Market Playback';
    const sectionDesc = document.querySelector('#live-market .section-desc');
    if (sectionDesc) sectionDesc.textContent = 'Watch the DQN agent trade on live market bars pulled from the backend. Green triangles = BUY, Red triangles = SELL.';
    document.querySelectorAll('.chart-title, .card-title').forEach((node) => {
      node.textContent = node.textContent
        .replace('Synthetic Price + Trade Signals', 'Live Price + Trade Signals')
        .replace('Synthetic Price Model', 'Live Market Feed');
    });
  } catch (error) {
    console.error(error);
    setBadge('ERROR', 'act-wait');
    const desc = document.querySelector('#live-market .section-desc');
    if (desc) desc.textContent = 'Unable to load live market data right now. The backend may be offline or the data source is unavailable.';
  }

  el('btnPlay')?.addEventListener('click', playPlayback);
  el('btnPause')?.addEventListener('click', stopPlayback);
  el('btnReset')?.addEventListener('click', () => {
    resetPlayback();
    if (STATE.payload) {
      updateFrame(0);
    }
  });

  document.querySelectorAll('.speed-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      STATE.speed = parseInt(btn.dataset.speed, 10);
    });
  });

  el('startSimBtn')?.addEventListener('click', (event) => {
    event.preventDefault();
    el('live-market')?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => playPlayback(), 500);
  });

  document.querySelectorAll('.glass-card, .kpi-card, .section-header').forEach((item) => {
    item.style.opacity = '0';
    item.style.transform = 'translateY(14px)';
    item.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
    requestAnimationFrame(() => {
      item.style.opacity = '1';
      item.style.transform = 'translateY(0)';
    });
  });
}

document.addEventListener('DOMContentLoaded', boot);
