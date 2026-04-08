'use strict';

const API_BASE = 'http://127.0.0.1:8000/api';
const WS_BASE = API_BASE.replace(/^http/, 'ws');
const DEFAULT_SYMBOL = 'AAPL';
const REFRESH_MS = 15000;

const state = {
  symbol: DEFAULT_SYMBOL,
  range: 'today',
  dashboard: null,
  latestQuote: null,
  priceChart: null,
  marketValueChart: null,
  marketValueSeries: [],
  allocationChart: null,
  refreshTimer: null,
  quoteSocket: null,
  reconnectTimer: null,
};

const dom = {};

function $(id) {
  return document.getElementById(id);
}

function formatCurrency(value) {
  const number = Number(value || 0);
  const sign = number < 0 ? '-' : '';
  return `${sign}$${Math.abs(number).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatPercent(value) {
  const number = Number(value || 0);
  const sign = number > 0 ? '+' : '';
  return `${sign}${number.toFixed(2)}%`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString('en-US');
}

function formatRatio(value) {
  const number = Number(value || 0);
  return `${number.toFixed(2)}x`;
}

function formatDate(value) {
  if (!value) return '-';
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  return dt.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }

  if (!response.ok) {
    const message = data?.detail || data?.message || response.statusText;
    throw new Error(message);
  }

  return data;
}

function setConnectionStatus(connected, message) {
  const status = dom.connectionStatus;
  if (!status) return;
  status.classList.toggle('is-live', connected);
  status.innerHTML = `<span class="status-dot"></span>${message}`;
}

function setActiveWatchSymbol(symbol) {
  document.querySelectorAll('.watch-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.symbol === symbol);
  });
}

function setActiveFilter(range) {
  document.querySelectorAll('.filter-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.range === range);
  });
}

function syncOrderForm() {
  const isLimit = dom.orderType.value === 'limit';
  dom.limitPrice.closest('label').classList.toggle('hidden', !isLimit);
  dom.tradePrice.readOnly = true;
  dom.tradePrice.value = state.latestQuote ? Number(state.latestQuote.current_price || 0).toFixed(2) : dom.tradePrice.value;
}

function buildPriceChart(history) {
  const labels = history.map((point) => point.label);
  const prices = history.map((point) => point.close);
  const volumes = history.map((point) => point.volume || 0);

  if (state.priceChart) {
    state.priceChart.data.labels = labels;
    state.priceChart.data.datasets[0].data = prices;
    state.priceChart.data.datasets[1].data = volumes;
    state.priceChart.update('none');
    return;
  }

  state.priceChart = new Chart(dom.priceChart, {
    data: {
      labels,
      datasets: [
        {
          type: 'line',
          label: 'Close',
          data: prices,
          borderColor: '#4de1c1',
          backgroundColor: 'rgba(77, 225, 193, 0.12)',
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          borderWidth: 2,
        },
        {
          type: 'bar',
          label: 'Volume',
          data: volumes,
          backgroundColor: 'rgba(246, 180, 77, 0.22)',
          borderWidth: 0,
          yAxisID: 'y1',
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#91a4bd', usePointStyle: true, boxWidth: 10 },
        },
        tooltip: {
          backgroundColor: 'rgba(7, 17, 29, 0.95)',
          titleColor: '#ecf4ff',
          bodyColor: '#ecf4ff',
          borderColor: 'rgba(148, 163, 184, 0.18)',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#91a4bd', maxTicksLimit: 6 },
        },
        y: {
          position: 'left',
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: {
            color: '#91a4bd',
            callback: (value) => `$${Number(value).toFixed(0)}`,
          },
        },
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: {
            color: '#91a4bd',
            callback: (value) => formatNumber(value),
          },
        },
      },
    },
  });
}

function buildAllocationChart(portfolio) {
  const positions = portfolio.filter((item) => Number(item.market_value) > 0);
  const labels = positions.map((item) => item.symbol);
  const values = positions.map((item) => Number(item.market_value));
  const palette = ['#4de1c1', '#f6b44d', '#fb7185', '#8b5cf6', '#60a5fa'];
  const colors = values.map((_, index) => palette[index % palette.length]);

  if (state.allocationChart) {
    state.allocationChart.data.labels = labels;
    state.allocationChart.data.datasets[0].data = values;
    state.allocationChart.data.datasets[0].backgroundColor = colors;
    state.allocationChart.update('none');
    return;
  }

  state.allocationChart = new Chart(dom.allocationChart, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: colors,
          borderWidth: 0,
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#91a4bd', boxWidth: 12, usePointStyle: true },
        },
      },
    },
  });
}

function buildMarketValueChart(appendSample = false) {
  const dataset = state.dashboard?.portfolio?.find((item) => item.symbol === state.symbol);
  const holdingQty = dataset ? Number(dataset.quantity || 0) : 0;
  const marketValue = state.latestQuote ? Number(state.latestQuote.current_price || 0) * holdingQty : 0;
  const price = state.latestQuote ? Number(state.latestQuote.current_price || 0) : 0;
  const nowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  if (!state.marketValueSeries) {
    state.marketValueSeries = [];
  }

  if (state.latestQuote && (appendSample || state.marketValueSeries.length === 0)) {
    state.marketValueSeries.push({
      label: nowLabel,
      price,
      value: marketValue,
    });
  }

  state.marketValueSeries = state.marketValueSeries.slice(-24);

  const labels = state.marketValueSeries.map((point) => point.label);
  const prices = state.marketValueSeries.map((point) => point.price);
  const values = state.marketValueSeries.map((point) => point.value);

  dom.marketValueLabel.textContent = `${state.symbol} position value`;

  if (state.marketValueChart) {
    state.marketValueChart.data.labels = labels;
    state.marketValueChart.data.datasets[0].data = prices;
    state.marketValueChart.data.datasets[1].data = values;
    state.marketValueChart.update('none');
    return;
  }

  state.marketValueChart = new Chart(dom.marketValueChart, {
    data: {
      labels,
      datasets: [
        {
          type: 'line',
          label: 'Market Price',
          data: prices,
          borderColor: '#4de1c1',
          backgroundColor: 'rgba(77, 225, 193, 0.10)',
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          borderWidth: 2,
        },
        {
          type: 'line',
          label: 'Position Value',
          data: values,
          borderColor: '#f6b44d',
          backgroundColor: 'rgba(246, 180, 77, 0.12)',
          pointRadius: 0,
          tension: 0.25,
          fill: false,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#91a4bd', usePointStyle: true, boxWidth: 10 },
        },
        tooltip: {
          backgroundColor: 'rgba(7, 17, 29, 0.95)',
          titleColor: '#ecf4ff',
          bodyColor: '#ecf4ff',
          borderColor: 'rgba(148, 163, 184, 0.18)',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#91a4bd', maxTicksLimit: 6 },
        },
        y: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: {
            color: '#91a4bd',
            callback: (value) => `$${Number(value).toFixed(0)}`,
          },
        },
      },
    },
  });
}

function renderQuote(quote, appendSample = false) {
  if (!quote) return;
  state.latestQuote = quote;

  dom.quoteSymbol.textContent = quote.symbol;
  dom.symbolInput.value = quote.symbol;
  dom.tradeSymbol.value = quote.symbol;
  dom.currentPrice.textContent = formatCurrency(quote.current_price);
  dom.priceChange.textContent = formatPercent(quote.change_pct);
  dom.priceChange.className = quote.change_pct >= 0 ? 'positive' : 'negative';
  dom.volumeValue.textContent = formatNumber(quote.volume);
  dom.trendValue.textContent = quote.trend;
  dom.marketBiasValue.textContent = quote.trend;
  dom.momentumValue.textContent = formatPercent(quote.momentum_5d || quote.change_pct);
  dom.rsiValue.textContent = Number(quote.rsi || 50).toFixed(1);
  dom.volumeRatioValue.textContent = formatRatio(quote.volume_ratio || 0);
  dom.tradePrice.value = Number(quote.current_price || 0).toFixed(2);

  buildPriceChart(quote.history || []);
  buildMarketValueChart(appendSample);
}

function renderAccount(dashboard) {
  const account = dashboard.account;
  const portfolio = dashboard.portfolio || [];
  const transactions = dashboard.transactions || [];
  const recommendation = dashboard.recommendation || {};

  dom.balanceValue.textContent = formatCurrency(account.balance);
  dom.portfolioValue.textContent = formatCurrency(account.portfolio_value);
  dom.pnlValue.textContent = formatCurrency(account.pnl);
  dom.pnlValue.className = account.pnl >= 0 ? 'positive' : 'negative';
  dom.holdingCount.textContent = `${portfolio.length}`;
  dom.edgeScoreValue.textContent = `${recommendation.edge_score ?? 50}`;

  dom.availableBalance.textContent = formatCurrency(account.balance);
  dom.totalInvested.textContent = formatCurrency(account.total_invested);
  dom.accountPortfolioValue.textContent = formatCurrency(account.portfolio_value);
  dom.accountPnL.textContent = formatCurrency(account.pnl);
  dom.accountPnL.className = account.pnl >= 0 ? 'positive' : 'negative';
  dom.portfolioCount.textContent = `${portfolio.length} position${portfolio.length === 1 ? '' : 's'}`;

  dom.portfolioTableBody.innerHTML = portfolio.length
    ? portfolio.map((row) => `
      <tr>
        <td>${row.symbol}</td>
        <td>${formatNumber(row.quantity)}</td>
        <td>${formatCurrency(row.avg_price)}</td>
        <td>${formatCurrency(row.current_price)}</td>
        <td>${formatCurrency(row.market_value)}</td>
        <td class="${row.pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(row.pnl)}</td>
      </tr>
    `).join('')
    : `
      <tr>
        <td colspan="6" class="muted">No open positions yet.</td>
      </tr>
    `;

  dom.transactionsTableBody.innerHTML = transactions.length
    ? transactions.map((row) => `
      <tr>
        <td>${formatDate(row.filled_at || row.timestamp)}</td>
        <td class="${row.type === 'BUY' ? 'positive' : row.type === 'SELL' ? 'negative' : ''}">${row.type}</td>
        <td>${row.order_type || 'MARKET'}</td>
        <td>${row.symbol}</td>
        <td>${formatNumber(row.quantity)}</td>
        <td>${formatCurrency(row.execution_price || row.price)}</td>
        <td>${formatCurrency(row.total)}</td>
        <td class="${row.status === 'pending' ? 'pending' : row.status === 'executed' ? 'positive' : ''}">${row.status}</td>
      </tr>
    `).join('')
    : `
      <tr>
        <td colspan="8" class="muted">No transactions recorded for this filter.</td>
      </tr>
    `;

  dom.transactionCount.textContent = `${transactions.length} record${transactions.length === 1 ? '' : 's'}`;

  dom.recommendationAction.textContent = recommendation.action || 'HOLD';
  dom.recommendationConfidence.textContent = `${Math.round((recommendation.confidence || 0) * 100)}%`;
  dom.recommendationEdge.textContent = `${recommendation.edge_score ?? 50}`;
  dom.recommendationRR.textContent = Number(recommendation.risk_reward || 1).toFixed(2);
  dom.recommendationTarget.textContent = formatCurrency(recommendation.target_price || 0);
  dom.recommendationStop.textContent = formatCurrency(recommendation.stop_price || 0);
  dom.recommendationRegime.textContent = recommendation.regime || 'Neutral / wait';
  dom.modelVoteValue.textContent = recommendation.dqn_action || '-';
  dom.recommendationReason.textContent = recommendation.reason || 'No recommendation available.';
  dom.executeAiTrade.disabled = !recommendation.action || recommendation.action === 'HOLD';
  dom.blueprintSummary.textContent = `${recommendation.action || 'WAIT'} | ${Math.round((recommendation.confidence || 0) * 100)}% confidence`;

  const factors = Array.isArray(recommendation.factors) ? recommendation.factors : [];
  dom.signalFactors.innerHTML = factors.length
    ? factors.map((factor) => `
      <div class="factor-item">
        <div class="factor-top">
          <span>${factor.name}</span>
          <strong class="${factor.direction === 'bullish' ? 'positive' : factor.direction === 'bearish' ? 'negative' : 'muted'}">${factor.direction || 'neutral'}</strong>
        </div>
        <div class="factor-meta">
          <span>${typeof factor.value === 'number' ? factor.value.toFixed(2) : factor.value}</span>
          <span>${factor.impact >= 0 ? '+' : ''}${Number(factor.impact || 0).toFixed(2)} pts</span>
        </div>
      </div>
    `).join('')
    : `<div class="factor-empty">No supporting factors available yet.</div>`;

  buildAllocationChart(portfolio);
}

function renderScenario(scenario) {
  if (!scenario) return;

  const signal = scenario.signal_snapshot || {};
  const account = scenario.account || {};
  const hypothetical = scenario.hypothetical_trade || {};

  dom.scenarioPrice.textContent = formatCurrency(scenario.shocked_price);
  dom.scenarioPortfolioDelta.textContent = formatCurrency(account.delta || 0);
  dom.scenarioHypotheticalPnl.textContent = formatCurrency(hypothetical.pnl_vs_now || 0);
  dom.scenarioAction.textContent = signal.action || 'HOLD';
  dom.scenarioConfidence.textContent = `${Math.round((signal.confidence || 0) * 100)}%`;
  dom.scenarioEdge.textContent = `${signal.edge_score ?? 50}`;
  dom.scenarioAccountValue.textContent = formatCurrency(account.portfolio_value_scenario || 0);
  dom.scenarioAction.className = signal.action === 'BUY' ? 'positive' : signal.action === 'SELL' ? 'negative' : '';

  const flipNote = scenario.signal_flip
    ? `Signal flips under the shock from the live baseline.`
    : `Signal remains aligned with the live baseline.`;
  dom.scenarioReason.textContent = `${signal.reason || 'No scenario signal available.'} ${flipNote}`;
}

async function loadScenario() {
  try {
    const move = Number(dom.scenarioMove.value || 0);
    const quantity = Number(dom.scenarioQuantity.value || 1);
    const scenario = await fetchJson(
      `/scenario?symbol=${encodeURIComponent(state.symbol)}&move_pct=${encodeURIComponent(move)}&quantity=${encodeURIComponent(quantity)}`,
    );
    renderScenario(scenario);
  } catch (error) {
    dom.scenarioReason.textContent = `Unable to run scenario: ${error.message}`;
  }
}

async function loadDashboard() {
  try {
    const dashboard = await fetchJson(`/dashboard?symbol=${encodeURIComponent(state.symbol)}&range=${encodeURIComponent(state.range)}`);
    state.dashboard = dashboard;
    renderAccount(dashboard);
    if (!state.latestQuote || state.latestQuote.symbol !== dashboard.quote.symbol) {
      renderQuote(dashboard.quote, false);
    }
    await loadScenario();
  } catch (error) {
    console.error(error);
    dom.tradeNote.textContent = `Unable to reach the backend: ${error.message}`;
  }
}

function closeQuoteSocket() {
  if (state.quoteSocket) {
    try {
      state.quoteSocket.onclose = null;
      state.quoteSocket.close();
    } catch {
      // ignore shutdown noise
    }
  }
  state.quoteSocket = null;
}

function connectQuoteSocket(symbol) {
  closeQuoteSocket();
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }

  const socket = new WebSocket(`${WS_BASE}/ws/quotes?symbol=${encodeURIComponent(symbol)}&interval=5`);
  state.quoteSocket = socket;

  socket.addEventListener('open', () => {
    setConnectionStatus(true, `Live feed connected | ${symbol}`);
  });

  socket.addEventListener('message', (event) => {
    const payload = JSON.parse(event.data);
    if (payload.error) {
      setConnectionStatus(false, `Feed error | ${payload.error}`);
      return;
    }
    if (payload.symbol === state.symbol) {
      renderQuote(payload, true);
    }
  });

  socket.addEventListener('close', () => {
    if (state.symbol !== symbol) return;
    setConnectionStatus(false, `Reconnecting feed | ${symbol}`);
    state.reconnectTimer = window.setTimeout(() => connectQuoteSocket(symbol), 2500);
  });

  socket.addEventListener('error', () => {
    setConnectionStatus(false, `Feed error | ${symbol}`);
  });
}

async function submitTrade(action) {
  const symbol = dom.tradeSymbol.value.trim().toUpperCase() || state.symbol;
  const quantity = Number(dom.tradeQuantity.value || 0);
  const price = Number(dom.tradePrice.value || 0);
  const orderType = dom.orderType.value;
  const limitPrice = Number(dom.limitPrice.value || 0);

  if (!symbol) {
    dom.tradeNote.textContent = 'Enter a symbol before placing an order.';
    return;
  }

  if (action !== 'hold' && (!Number.isFinite(quantity) || quantity <= 0)) {
    dom.tradeNote.textContent = 'Quantity must be at least 1 for buy and sell orders.';
    return;
  }

  if (orderType === 'limit' && (!Number.isFinite(limitPrice) || limitPrice <= 0)) {
    dom.tradeNote.textContent = 'Limit price is required for limit orders.';
    return;
  }

  try {
    const response = await fetchJson(`/trade/${action}`, {
      method: 'POST',
      body: JSON.stringify({
        user_id: 1,
        symbol,
        quantity: action === 'hold' ? 1 : quantity,
        price: orderType === 'market' ? price : limitPrice,
        order_type: orderType,
        limit_price: orderType === 'limit' ? limitPrice : null,
      }),
    });

    dom.tradeNote.textContent = response.message || `${action.toUpperCase()} order completed for ${symbol}.`;
    state.symbol = symbol;
    setActiveWatchSymbol(symbol);
    connectQuoteSocket(symbol);
    await loadDashboard();
  } catch (error) {
    dom.tradeNote.textContent = error.message;
  }
}

async function executeAiTrade() {
  if (!state.dashboard?.recommendation?.action || state.dashboard.recommendation.action === 'HOLD') {
    return;
  }

  await submitTrade(state.dashboard.recommendation.action.toLowerCase());
}

function setSymbol(symbol) {
  const nextSymbol = symbol.trim().toUpperCase();
  if (!nextSymbol) return;
  state.symbol = nextSymbol;
  state.marketValueSeries = [];
  dom.symbolInput.value = nextSymbol;
  dom.tradeSymbol.value = nextSymbol;
  setActiveWatchSymbol(nextSymbol);
  connectQuoteSocket(nextSymbol);
  loadDashboard();
}

function setRange(range) {
  state.range = range;
  setActiveFilter(range);
  loadDashboard();
}

function bindEvents() {
  document.querySelectorAll('.watch-btn').forEach((button) => {
    button.addEventListener('click', () => setSymbol(button.dataset.symbol));
  });

  dom.symbolInput.addEventListener('change', () => setSymbol(dom.symbolInput.value));
  dom.symbolInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      setSymbol(dom.symbolInput.value);
    }
  });

  dom.tradeSymbol.addEventListener('change', () => setSymbol(dom.tradeSymbol.value));
  dom.tradeSymbol.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      setSymbol(dom.tradeSymbol.value);
    }
  });

  dom.scenarioMove.addEventListener('change', loadScenario);
  dom.scenarioQuantity.addEventListener('change', loadScenario);
  dom.runScenarioBtn.addEventListener('click', loadScenario);

  dom.orderType.addEventListener('change', () => {
    syncOrderForm();
  });

  document.querySelectorAll('.trade-btn').forEach((button) => {
    button.addEventListener('click', () => {
      submitTrade(button.dataset.action);
    });
  });

  document.querySelectorAll('.filter-btn').forEach((button) => {
    button.addEventListener('click', () => setRange(button.dataset.range));
  });

  dom.executeAiTrade.addEventListener('click', executeAiTrade);
}

function cacheDom() {
  dom.connectionStatus = $('connectionStatus');
  dom.quoteSymbol = $('quoteSymbol');
  dom.symbolInput = $('symbolInput');
  dom.currentPrice = $('currentPrice');
  dom.priceChange = $('priceChange');
  dom.volumeValue = $('volumeValue');
  dom.trendValue = $('trendValue');
  dom.marketBiasValue = $('marketBiasValue');
  dom.momentumValue = $('momentumValue');
  dom.rsiValue = $('rsiValue');
  dom.volumeRatioValue = $('volumeRatioValue');
  dom.priceChart = $('priceChart');
  dom.tradeSymbol = $('tradeSymbol');
  dom.orderType = $('orderType');
  dom.limitPrice = $('limitPrice');
  dom.tradePrice = $('tradePrice');
  dom.tradeQuantity = $('tradeQuantity');
  dom.tradeNote = $('tradeNote');
  dom.recommendationAction = $('recommendationAction');
  dom.recommendationConfidence = $('recommendationConfidence');
  dom.recommendationEdge = $('recommendationEdge');
  dom.recommendationRR = $('recommendationRR');
  dom.recommendationTarget = $('recommendationTarget');
  dom.recommendationStop = $('recommendationStop');
  dom.recommendationRegime = $('recommendationRegime');
  dom.modelVoteValue = $('modelVoteValue');
  dom.recommendationReason = $('recommendationReason');
  dom.signalFactors = $('signalFactors');
  dom.blueprintSummary = $('blueprintSummary');
  dom.executeAiTrade = $('executeAiTrade');
  dom.balanceValue = $('balanceValue');
  dom.portfolioValue = $('portfolioValue');
  dom.pnlValue = $('pnlValue');
  dom.holdingCount = $('holdingCount');
  dom.edgeScoreValue = $('edgeScoreValue');
  dom.availableBalance = $('availableBalance');
  dom.totalInvested = $('totalInvested');
  dom.accountPortfolioValue = $('accountPortfolioValue');
  dom.accountPnL = $('accountPnL');
  dom.portfolioCount = $('portfolioCount');
  dom.portfolioTableBody = $('portfolioTableBody');
  dom.allocationChart = $('allocationChart');
  dom.marketValueChart = $('marketValueChart');
  dom.marketValueLabel = $('marketValueLabel');
  dom.scenarioMove = $('scenarioMove');
  dom.scenarioQuantity = $('scenarioQuantity');
  dom.scenarioPrice = $('scenarioPrice');
  dom.scenarioPortfolioDelta = $('scenarioPortfolioDelta');
  dom.scenarioHypotheticalPnl = $('scenarioHypotheticalPnl');
  dom.scenarioAction = $('scenarioAction');
  dom.scenarioConfidence = $('scenarioConfidence');
  dom.scenarioEdge = $('scenarioEdge');
  dom.scenarioAccountValue = $('scenarioAccountValue');
  dom.scenarioReason = $('scenarioReason');
  dom.runScenarioBtn = $('runScenarioBtn');
  dom.transactionCount = $('transactionCount');
  dom.transactionsTableBody = $('transactionsTableBody');
}

function initNavGlow() {
  const nav = $('navbar');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  }, { passive: true });
}

document.addEventListener('DOMContentLoaded', async () => {
  cacheDom();
  initNavGlow();
  bindEvents();
  setActiveWatchSymbol(state.symbol);
  setActiveFilter(state.range);
  syncOrderForm();
  setConnectionStatus(false, 'Connecting to market feed');
  connectQuoteSocket(state.symbol);
  await loadDashboard();
  state.refreshTimer = window.setInterval(loadDashboard, REFRESH_MS);
});
