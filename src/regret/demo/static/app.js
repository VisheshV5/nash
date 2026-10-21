// Regret demo table: talks to /demo/api and renders the table, controls, decision clock and the
// reasoning dialog. Everything the dialog says comes from numbers the agent computed.
"use strict";

const $ = (id) => document.getElementById(id);
const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
const DEAL_STEP_MS = 140; // between cards of the flop
const RUNOUT_STEP_MS = 700; // between turn and river when the chips went in early

let session = null;
let mode = "play";
let autoTimer = null;
let pending = false;
// What the table showed last, so only new cards and new bets animate.
let shown = { hand: 0, board: 0, agentCards: false, bets: [0, 0] };
let resultTimer = null;
const clock = { seconds: 30, key: null, deadline: 0, tick: null };

// ------------------------------------------------------------------ server

async function api(path, body) {
  const res = await fetch(`api/${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `request failed (${res.status})`);
  return data;
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}

async function run(fn) {
  showError("");
  pending = true;
  stopClock();
  if (session) renderTable(session);
  try {
    session = await fn();
  } catch (e) {
    showError(e.message);
    stopAuto();
  } finally {
    pending = false;
  }
  if (session) render();
}

// ------------------------------------------------------------------ cards and words

function cardEl(card, opts = {}) {
  const el = document.createElement("div");
  el.className = "card" + (opts.mini ? " mini" : "");
  if (!card) {
    el.classList.add("back");
    el.setAttribute("aria-label", "face-down card");
    return el;
  }
  const rank = card[0] === "T" ? "10" : card[0];
  if (card[1] === "h" || card[1] === "d") el.classList.add("red");
  el.innerHTML = `${rank}<span class="s">${SUITS[card[1]]}</span>`;
  el.setAttribute("aria-label", card);
  return el;
}

function cardsEl(cards, opts) {
  const wrap = document.createElement("span");
  wrap.className = "cards";
  cards.forEach((c) => wrap.appendChild(cardEl(c, opts)));
  return wrap;
}

const cardText = (c) => (c[0] === "T" ? "10" : c[0]) + SUITS[c[1]];
const pct = (p) => `${Math.round(p * 100)}%`;
const bb = (x) => `${+(+x).toFixed(2)} bb`;

function potWords(f) {
  const k = Math.round(f * 100);
  const named = { 33: "⅓ pot", 50: "½ pot", 66: "⅔ pot", 67: "⅔ pot", 75: "¾ pot", 100: "the pot", 125: "1¼ pots", 150: "1½ pots", 200: "2 pots" };
  return named[k] || `${k}% of the pot`;
}

// Strategy keys from the agent (fold, check, call, bet_0.75pot, raise_3x, raise_2.5bb, all_in).
function labelOf(key, e = {}) {
  if (key === "fold") return "Fold";
  if (key === "check") return "Check";
  if (key === "call") return e.to_call ? `Call ${bb(e.to_call)}` : "Call";
  if (key === "all_in") return "All-in";
  let m = key.match(/^bet_([\d.]+)pot$/);
  if (m) return `Bet ${potWords(parseFloat(m[1]))}`;
  m = key.match(/^raise_([\d.]+)x$/);
  if (m) return `Raise to ${m[1]}× the bet`;
  m = key.match(/^raise_([\d.]+)bb$/);
  if (m) return `Raise to ${m[1]} bb`;
  return key;
}

function headline(d) {
  const r = d.recommended;
  const e = d.explain || {};
  const key = r.label || r.action;
  if (key === "all_in") return `Went all-in (${bb(r.amount)})`;
  if (r.action === "fold") return "Folded";
  if (r.action === "check") return "Checked";
  if (r.action === "call") return `Called ${bb(e.to_call)}`;
  const m = (r.label || "").match(/^bet_([\d.]+)pot$/);
  if (r.action === "bet") return `Bet ${bb(r.amount)}` + (m ? ` (${potWords(parseFloat(m[1]))})` : "");
  return `Raised to ${bb(r.amount)}`;
}

const lower = (s) => s.charAt(0).toLowerCase() + s.slice(1);

// One or two plain sentences on how often it makes this play, including when it mixed.
function frequencyText(d) {
  const e = d.explain || {};
  const entries = Object.entries(d.strategy).sort((a, b) => b[1] - a[1]);
  const key = d.recommended.label || d.recommended.action;
  const p = d.strategy[key] ?? 0;
  const [topKey, topP] = entries[0];
  if (p >= 0.995) return "It always makes this play here with this hand.";
  if (p >= 0.9) return `It almost always makes this play here with this hand (${pct(p)}).`;
  if (topKey === key) {
    const [nextKey, nextP] = entries[1] || [];
    return nextKey
      ? `Its most common play here (${pct(p)}). The rest of the time: ${lower(labelOf(nextKey, e))} (${pct(nextP)}).`
      : `Its only play here.`;
  }
  return `A mixed spot: it usually would ${lower(labelOf(topKey, e))} (${pct(topP)}), and makes this play ${pct(p)} of the time. ` +
    `Mixing keeps its bet sizes from giving its hand away.`;
}

function factRows(d) {
  const e = d.explain || {};
  const rows = [];
  if (e.equity_vs_random_hand != null) {
    rows.push(["Hand strength",
      `${d.hole.map(cardText).join(" ")} would win ${pct(e.equity_vs_random_hand)} against a random hand from here. ` +
      `A rough guide: it doesn't know your cards.`]);
  }
  if (e.to_call > 0) {
    const be = e.to_call / (e.pot + e.to_call);
    rows.push(["Price", `${bb(e.to_call)} to call into ${bb(e.pot)}: a call breaks even if it wins ${pct(be)}.`]);
  }
  if (d.source === "subgame_solve" && e.search) {
    const s = e.search;
    const off = (s.exploitability_chips / 100) / Math.max(e.pot, 0.01);
    rows.push(["How it decided",
      `Re-solved this ${e.street} live: ${s.iterations.toLocaleString()} rounds of self-play in ` +
      `${(s.solve_ms / 1000).toFixed(2)} s, ending within ${off < 0.001 ? "0.1%" : (off * 100).toFixed(1) + "%"} ` +
      `of the pot of perfect play for this spot.`]);
  } else {
    rows.push(["How it decided", `Looked up its trained strategy for this spot (${Math.round(d.latency_ms)} ms).`]);
  }
  const mapped = [...new Set((e.abstract_histories || []).flatMap((h) => h.translations))];
  for (const t of mapped) {
    const m = t.match(/^(\S+) -> (\S+) \((\d+)%\)$/);
    if (m) {
      rows.push(["Your bet size",
        `It never trained on “${lower(labelOf(m[1]))}”, so it read it as “${lower(labelOf(m[2]))}”` +
        (m[3] !== "100" ? ` (${m[3]}% weight; the rest on the next size).` : ".")]);
    }
  }
  if (e.fallback_share > 0) {
    rows.push(["Off its map", `${pct(e.fallback_share)} of this decision fell back to check/call: the betting went past what it trained on.`]);
  }
  return rows;
}

// ------------------------------------------------------------------ render

function render() {
  const s = session;
  renderTable(s);
  renderControls(s);
  renderLog(s.log);
  renderReasoningButton(s);
  if ($("reasoning").open) renderDecisions(s);
}

function renderTable(s) {
  $("hand-no").textContent = `Hand ${s.hand}`;
  $("net").textContent = (s.agent_net_bb > 0 ? "+" : "") + s.agent_net_bb;
  const newHand = s.hand !== shown.hand;
  if (newHand) shown = { hand: s.hand, board: 0, agentCards: false, bets: [0, 0] };

  const human = 1 - s.agent_seat;
  const bottom = s.mode === "play" ? human : s.agent_seat;
  for (const [el, seat] of [[$("seat-bottom"), bottom], [$("seat-top"), 1 - bottom]]) {
    renderSeat(el, s, seat, newHand);
  }

  $("pot").textContent = `Pot ${bb(s.pot_bb)}`;
  $("street").textContent = s.result ? (s.result.showdown ? "Showdown" : "Hand over") : s.street;

  // Board: new cards deal in; an early all-in runs the rest out one card at a time.
  const cards = s.result ? s.result.board : s.board;
  const board = $("board");
  const before = Math.min(shown.board, cards.length);
  const runout = s.result && cards.length - before > 0 && before >= 3;
  const step = runout ? RUNOUT_STEP_MS : DEAL_STEP_MS;
  board.innerHTML = "";
  cards.forEach((c, i) => {
    const el = cardEl(c);
    if (i >= before) {
      el.classList.add("deal");
      el.style.animationDelay = `${(i - before) * step}ms`;
    }
    board.appendChild(el);
  });
  for (let i = cards.length; i < 5; i++) {
    const slot = document.createElement("div");
    slot.className = "card slot";
    board.appendChild(slot);
  }
  const revealMs = cards.length > before ? (cards.length - before - 1) * step + 450 : 0;
  shown.board = cards.length;
  renderResult(s, s.result ? (runout ? revealMs : 150) : 0);
}

function lastAction(s, seat) {
  for (let i = s.log.length - 1; i >= 0; i--) {
    const l = s.log[i];
    if (l.seat === seat) return l.street === s.street || s.result ? l.text : null;
  }
  return null;
}

function renderSeat(el, s, seat, newHand) {
  const info = s.seats[seat];
  const isAgent = seat === s.agent_seat;
  el.innerHTML = "";
  el.classList.toggle("acting", !pending && s.to_act === seat && !s.result);
  el.classList.toggle("folded", info.status === "folded");

  const who = document.createElement("div");
  who.className = "who";
  who.innerHTML = `<span class="name">${info.name} · ${info.position === "BTN" ? "Button" : "Big blind"}</span>` +
    `<span class="meta">Stack ${bb(info.stack_bb)}${info.status === "all_in" ? " · all-in" : ""}</span>`;

  const mid = document.createElement("div");
  mid.className = "seat-mid";
  if (isAgent && pending && s.mode === "play") {
    mid.innerHTML = `<span class="thinking">Thinking <i></i><i></i><i></i></span>`;
  } else {
    const last = lastAction(s, seat);
    if (last) mid.innerHTML = `<span class="last">${last.charAt(0).toUpperCase() + last.slice(1)}</span>`;
  }
  if (info.bet_bb > 0) {
    const chip = document.createElement("span");
    chip.className = "betchip" + (info.bet_bb !== shown.bets[seat] ? " pop" : "");
    chip.textContent = bb(info.bet_bb);
    mid.appendChild(chip);
  }
  shown.bets[seat] = info.bet_bb;

  const cards = document.createElement("div");
  cards.className = "cards";
  const visible = !!info.cards;
  (info.cards || [null, null]).forEach((c, i) => {
    const card = cardEl(c);
    if (newHand) {
      card.classList.add("deal");
      card.style.animationDelay = `${i * DEAL_STEP_MS + (isAgent ? 0 : 2 * DEAL_STEP_MS)}ms`;
    } else if (isAgent && visible && !shown.agentCards) {
      card.classList.add("flip");
      card.style.animationDelay = `${i * 120}ms`;
    }
    cards.appendChild(card);
  });
  if (isAgent) shown.agentCards = visible;
  el.append(who, mid, cards);
}

function renderControls(s) {
  $("controls-play").hidden = s.mode !== "play" || !!s.result;
  $("controls-watch").hidden = s.mode !== "watch" || !!s.result;
  $("step").disabled = pending || !!s.result;
  const actions = $("actions");
  actions.innerHTML = "";
  $("sizer").hidden = true;
  const legal = s.legal;
  if (!legal || pending) {
    stopClock();
    return;
  }
  const add = (text, fn, primary) => {
    const b = document.createElement("button");
    b.textContent = text;
    if (primary) b.className = "primary";
    b.addEventListener("click", fn);
    actions.appendChild(b);
    return b;
  };
  if (legal.fold) add("Fold", () => act("fold"));
  add(legal.call_bb ? `Call ${bb(legal.call_bb)}` : "Check", () => act(legal.call_bb ? "call" : "check"));
  if (legal.can_raise) {
    const kind = legal.current_bet_bb > 0 ? "raise" : "bet";
    const size = $("size");
    size.min = legal.min_raise_to_bb;
    size.max = legal.max_raise_to_bb;
    size.value = Math.min(legal.max_raise_to_bb, Math.max(legal.min_raise_to_bb,
      legal.current_bet_bb + (legal.pot_bb + legal.call_bb) * 0.75));
    const label = () => `${kind === "bet" ? "Bet" : "Raise to"} ${bb(parseFloat(size.value).toFixed(1))}`;
    const sizeBtn = add(label(), () => act(kind, parseFloat(size.value)), true);
    size.oninput = () => { sizeBtn.textContent = label(); };
    add(`All-in ${bb(legal.max_raise_to_bb)}`, () => act("all_in"));
    const presets = $("presets");
    presets.innerHTML = "";
    for (const [name, f] of [["½ pot", 0.5], ["¾ pot", 0.75], ["Pot", 1], ["2× pot", 2]]) {
      const to = legal.current_bet_bb + (legal.pot_bb + legal.call_bb) * f;
      if (to < legal.min_raise_to_bb || to > legal.max_raise_to_bb) continue;
      const b = document.createElement("button");
      b.textContent = name;
      b.addEventListener("click", () => { size.value = to; size.oninput(); });
      presets.appendChild(b);
    }
    $("sizer").hidden = false;
  }
  startClock(s, legal);
}

function renderResult(s, delayMs) {
  clearTimeout(resultTimer);
  const el = $("result");
  if (!s.result) {
    el.hidden = true;
    return;
  }
  const agentWon = s.result.payoffs_bb[s.agent_seat];
  const other = s.seats[1 - s.agent_seat].name;
  let title;
  let good;
  if (s.mode === "play") {
    title = agentWon < 0 ? `You win ${bb(-agentWon)}` : agentWon > 0 ? `Regret wins ${bb(agentWon)}` : "Split pot";
    good = agentWon < 0;
  } else {
    title = agentWon > 0 ? `Regret wins ${bb(agentWon)}` : agentWon < 0 ? `${other} wins ${bb(-agentWon)}` : "Split pot";
    good = agentWon > 0;
  }
  let detail;
  if (s.result.showdown && s.result.hands) {
    const mine = s.mode === "play" ? "You" : other;
    detail = `Showdown. ${mine}: ${s.result.hands[1 - s.agent_seat]}. Regret: ${s.result.hands[s.agent_seat]}.`;
  } else {
    const fold = [...s.log].reverse().find((l) => l.text === "folds");
    detail = fold ? `${fold.who} folded.` : "";
  }
  $("result-title").textContent = title;
  $("result-detail").textContent = detail;
  el.classList.toggle("win", agentWon !== 0 && good);
  el.classList.toggle("lose", agentWon !== 0 && !good);
  $("result-why").hidden = s.decisions.length === 0;
  const show = () => {
    el.hidden = false;
    if (s.mode === "play") $("next").focus({ preventScroll: true });
  };
  if (delayMs > 0) {
    el.hidden = true;
    resultTimer = setTimeout(show, delayMs);
  } else {
    show();
  }
}

function renderLog(log) {
  const el = $("log");
  el.innerHTML = log.map((l) => `<li><span class="st">${l.street}</span>${l.who} ${l.text}</li>`).join("");
  el.scrollTop = el.scrollHeight;
}

// ------------------------------------------------------------------ reasoning dialog

function renderReasoningButton(s) {
  const btn = $("open-reasoning");
  const n = s.decisions.length;
  btn.disabled = n === 0;
  btn.title = s.decisions_hidden
    ? "Shown when the hand ends: it would give away Regret's cards"
    : n === 0 ? "Regret hasn't acted yet this hand" : "";
  $("reasoning-count").hidden = n === 0;
  $("reasoning-count").textContent = n;
}

function decisionItem(d) {
  const li = document.createElement("li");
  li.className = "step";
  const e = d.explain || {};
  const chosen = d.recommended.label || d.recommended.action;

  const head = document.createElement("div");
  head.className = "step-head";
  head.innerHTML = `<span class="street-tag">${d.street}</span>`;
  const holding = document.createElement("span");
  holding.className = "hand-row";
  holding.append("Regret held", cardsEl(d.hole, { mini: true }));
  head.appendChild(holding);
  if (d.board && d.board.length) {
    const board = document.createElement("span");
    board.className = "hand-row";
    board.append("Board", cardsEl(d.board, { mini: true }));
    head.appendChild(board);
  }

  const h = document.createElement("h3");
  h.textContent = headline(d);
  const why = document.createElement("p");
  why.textContent = frequencyText(d);

  const bars = document.createElement("div");
  bars.className = "bars";
  bars.setAttribute("aria-label", "How often it takes each action here");
  Object.entries(d.strategy).sort((a, b) => b[1] - a[1]).filter(([, p]) => p >= 0.005).forEach(([k, p]) => {
    const row = document.createElement("div");
    row.className = "brow" + (k === chosen ? " chosen" : "");
    row.innerHTML = `<span class="label"></span><span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
      `<span class="pct">${pct(p)}</span>`;
    row.querySelector(".label").textContent = labelOf(k, e);
    bars.appendChild(row);
  });

  const facts = document.createElement("ul");
  facts.className = "facts";
  for (const [k, v] of factRows(d)) {
    const row = document.createElement("li");
    row.innerHTML = `<span class="k"></span><span></span>`;
    row.children[0].textContent = k;
    row.children[1].textContent = v;
    facts.appendChild(row);
  }

  const tech = document.createElement("details");
  tech.className = "tech";
  tech.innerHTML = `<summary>Technical details</summary>`;
  const techText = document.createElement("p");
  techText.textContent = `Hand group ${e.bucket} of ${e.buckets_on_street} on the ${e.street} · ` +
    `${d.source === "subgame_solve" ? `live solve, ${e.search?.nodes} tree nodes` : "blueprint lookup"} · ` +
    `${Math.round(d.latency_ms)} ms`;
  tech.appendChild(techText);

  li.append(head, h, why, bars, facts, tech);
  return li;
}

function renderDecisions(s) {
  const list = $("decisions");
  list.innerHTML = "";
  s.decisions.forEach((d) => list.appendChild(decisionItem(d)));
  $("reasoning-title").textContent = `How Regret played hand ${s.hand}`;
  const empty = $("reasoning-empty");
  empty.hidden = s.decisions.length > 0;
  empty.textContent = s.decisions_hidden
    ? "Regret's reasoning shows when the hand ends, since it would give away its cards."
    : "Regret hasn't made a decision yet this hand.";
}

function openReasoning() {
  if (!session || session.decisions.length === 0) return;
  renderDecisions(session);
  $("reasoning").showModal();
}

// ------------------------------------------------------------------ decision clock

function startClock(s, legal) {
  const key = `${s.hand}:${s.log.length}`;
  if (!clock.seconds) {
    stopClock();
    return;
  }
  if (clock.key !== key) {
    stopClock();
    clock.key = key;
    clock.deadline = Date.now() + clock.seconds * 1000;
  }
  $("timer").hidden = false;
  const tick = () => {
    const left = Math.max(0, clock.deadline - Date.now());
    const frac = left / (clock.seconds * 1000);
    $("timer-fill").style.transform = `scaleX(${frac})`;
    $("timer-text").textContent = `${Math.ceil(left / 1000)} s`;
    $("timer").classList.toggle("warn", left <= 10_000 && left > 5_000);
    $("timer").classList.toggle("danger", left <= 5_000);
    if (left <= 0) {
      stopClock();
      act(legal.call_bb ? "fold" : "check"); // out of time: check if free, else fold
    }
  };
  clearInterval(clock.tick);
  clock.tick = setInterval(tick, 250);
  tick();
}

function stopClock() {
  clearInterval(clock.tick);
  clock.tick = null;
  clock.key = null;
  $("timer").hidden = true;
}

// ------------------------------------------------------------------ actions

function act(kind, amount) {
  if (pending || !session) return;
  run(() => api(`sessions/${session.id}/act`, { kind, amount_bb: amount ?? null }));
}

function nextHand() {
  if (pending || !session || !session.result) return;
  run(() => api(`sessions/${session.id}/next`));
}

function stopAuto() {
  clearInterval(autoTimer);
  autoTimer = null;
  $("auto").textContent = "Auto-play";
}

function start(newMode) {
  stopAuto();
  mode = newMode;
  $("mode-play").setAttribute("aria-selected", String(mode === "play"));
  $("mode-watch").setAttribute("aria-selected", String(mode === "watch"));
  shown.hand = 0;
  run(() => api("sessions", { mode, opponent: $("opponent").value }));
}

$("mode-play").addEventListener("click", () => start("play"));
$("mode-watch").addEventListener("click", () => start("watch"));
$("opponent").addEventListener("change", () => start("watch"));
$("step").addEventListener("click", () => run(() => api(`sessions/${session.id}/step`)));
$("next").addEventListener("click", nextHand);
$("result-why").addEventListener("click", openReasoning);
$("open-reasoning").addEventListener("click", openReasoning);
$("close-reasoning").addEventListener("click", () => $("reasoning").close());
$("reasoning").addEventListener("click", (e) => {
  if (e.target === $("reasoning")) $("reasoning").close(); // backdrop
});
$("auto").addEventListener("click", () => {
  if (autoTimer) return stopAuto();
  $("auto").textContent = "Pause";
  autoTimer = setInterval(() => {
    if (pending || !session) return;
    if (session.result && $("result").hidden) return; // let the runout finish
    run(() => api(`sessions/${session.id}/${session.result ? "next" : "step"}`));
  }, 1000);
});

try {
  const saved = localStorage.getItem("regret.clock");
  if (saved !== null) $("clock").value = saved;
} catch { /* storage unavailable: keep the default */ }
clock.seconds = parseInt($("clock").value, 10);
$("clock").addEventListener("change", () => {
  clock.seconds = parseInt($("clock").value, 10);
  try { localStorage.setItem("regret.clock", $("clock").value); } catch { /* ignore */ }
  stopClock();
  if (session) renderControls(session);
});

document.addEventListener("keydown", (e) => {
  if ($("reasoning").open || e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if ((e.key === " " || e.key === "Enter") && session?.result && !$("result").hidden && tag !== "BUTTON") {
    e.preventDefault();
    nextHand();
  } else if (e.key === "r" || e.key === "R") {
    openReasoning();
  }
});

start("play");
