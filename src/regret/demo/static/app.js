// Regret demo table: talks to /demo/api and renders the table, controls and decision panel.
"use strict";

const $ = (id) => document.getElementById(id);
const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
const LABELS = { fold: "Fold", check: "Check", call: "Call", all_in: "All-in" };
let session = null;
let mode = "play";
let autoTimer = null;

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
  const el = $("error");
  el.textContent = message || "";
  el.hidden = !message;
}

async function run(fn) {
  showError("");
  try {
    session = await fn();
    render();
  } catch (e) {
    showError(e.message);
    stopAuto();
  }
}

function cardEl(card) {
  const el = document.createElement("div");
  if (!card) {
    el.className = "card back";
    el.setAttribute("aria-label", "hidden card");
    return el;
  }
  const rank = card[0] === "T" ? "10" : card[0];
  const suit = card[1];
  el.className = "card" + (suit === "h" || suit === "d" ? " red" : "");
  el.innerHTML = `${rank}<span class="s">${SUITS[suit]}</span>`;
  el.setAttribute("aria-label", card);
  return el;
}

function labelOf(key) {
  if (LABELS[key]) return LABELS[key];
  let m = key.match(/^bet_([\d.]+)pot$/);
  if (m) return `Bet ${Math.round(parseFloat(m[1]) * 100)}% pot`;
  m = key.match(/^raise_([\d.]+)x$/);
  if (m) return `Raise ${m[1]}x`;
  m = key.match(/^raise_([\d.]+)bb$/);
  if (m) return `Raise to ${m[1]}`;
  return key;
}

function renderSeat(el, seat, acting) {
  el.innerHTML = "";
  el.className = el.className.replace(/ acting| folded/g, "") + (acting ? " acting" : "") +
    (seat.status === "folded" ? " folded" : "");
  const who = document.createElement("div");
  who.className = "who";
  who.innerHTML = `<span class="name">${seat.name} · ${seat.position}</span>` +
    `<span class="meta">Stack ${seat.stack_bb} bb${seat.status === "all_in" ? " · all-in" : ""}</span>`;
  const cards = document.createElement("div");
  cards.className = "cards";
  (seat.cards || [null, null]).forEach((c) => cards.appendChild(cardEl(c)));
  el.append(who);
  if (seat.bet_bb > 0) {
    const chip = document.createElement("span");
    chip.className = "betchip";
    chip.textContent = `${seat.bet_bb} bb`;
    el.append(chip);
  }
  el.append(cards);
}

function render() {
  const s = session;
  $("hand-no").textContent = `Hand ${s.hand}`;
  const net = s.agent_net_bb;
  $("net").textContent = (net > 0 ? "+" : "") + net;
  const human = 1 - s.agent_seat;
  // In play mode you sit at the bottom; in watch mode Regret does.
  const bottom = s.mode === "play" ? human : s.agent_seat;
  renderSeat($("seat-bottom"), s.seats[bottom], s.to_act === bottom);
  renderSeat($("seat-top"), s.seats[1 - bottom], s.to_act === 1 - bottom);

  $("pot").textContent = `Pot ${s.pot_bb} bb`;
  $("street").textContent = s.result ? "Hand over" : s.street;
  const board = $("board");
  board.innerHTML = "";
  const shown = s.result ? s.result.board : s.board;
  shown.forEach((c) => board.appendChild(cardEl(c)));
  for (let i = shown.length; i < 5; i++) {
    const slot = document.createElement("div");
    slot.className = "card slot";
    board.appendChild(slot);
  }

  renderControls(s);
  renderDecisions(s);
  renderLog(s.log);
  renderResult(s);
}

function renderControls(s) {
  $("controls-play").hidden = s.mode !== "play";
  $("controls-watch").hidden = s.mode !== "watch";
  $("step").disabled = !!s.result;
  const actions = $("actions");
  actions.innerHTML = "";
  $("sizer").hidden = true;
  const legal = s.legal;
  if (!legal) return;
  const add = (text, fn, primary) => {
    const b = document.createElement("button");
    b.textContent = text;
    if (primary) b.className = "primary";
    b.addEventListener("click", fn);
    actions.appendChild(b);
  };
  if (legal.fold) add("Fold", () => act("fold"));
  add(legal.call_bb ? `Call ${legal.call_bb}` : "Check", () => act(legal.call_bb ? "call" : "check"));
  if (legal.can_raise) {
    const kind = legal.current_bet_bb > 0 ? "raise" : "bet";
    const size = $("size");
    size.min = legal.min_raise_to_bb;
    size.max = legal.max_raise_to_bb;
    size.value = Math.min(legal.max_raise_to_bb, Math.max(legal.min_raise_to_bb,
      legal.current_bet_bb + (legal.pot_bb + legal.call_bb) * 0.75));
    const label = () => `${kind === "bet" ? "Bet" : "Raise to"} ${parseFloat(size.value).toFixed(1)}`;
    add(label(), () => act(kind, parseFloat(size.value)), true);
    const sizeBtn = actions.lastChild;
    size.oninput = () => { sizeBtn.textContent = label(); };
    add(`All-in ${legal.max_raise_to_bb}`, () => act("all_in"));
    const presets = $("presets");
    presets.innerHTML = "";
    for (const [name, f] of [["½ pot", 0.5], ["¾ pot", 0.75], ["Pot", 1], ["2x pot", 2]]) {
      const to = legal.current_bet_bb + (legal.pot_bb + legal.call_bb) * f;
      if (to < legal.min_raise_to_bb || to > legal.max_raise_to_bb) continue;
      const b = document.createElement("button");
      b.textContent = name;
      b.addEventListener("click", () => { size.value = to; size.oninput(); });
      presets.appendChild(b);
    }
    $("sizer").hidden = false;
  }
}

function decisionBlock(d) {
  const block = document.createElement("div");
  block.className = "dblock";
  const chosen = d.recommended.label || d.recommended.action;
  const e = d.explain || {};
  const bars = Object.entries(d.strategy).sort((a, b) => b[1] - a[1]).map(([k, p]) =>
    `<div class="brow${k === chosen ? " chosen" : ""}"><span class="label" title="${k}">${labelOf(k)}</span>` +
    `<span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
    `<span class="pct">${Math.round(p * 100)}%</span></div>`).join("");
  const facts = [
    ["Equity vs a random hand", e.equity_vs_random_hand != null ? `${Math.round(e.equity_vs_random_hand * 100)}%` : "—"],
    ["Pot / to call", `${e.pot} / ${e.to_call} bb`],
    ["Hand group", `${e.bucket} of ${e.buckets_on_street} on the ${e.street}`],
    ["Decided in", `${d.latency_ms} ms`],
  ];
  if (e.search) {
    facts.push(["Solver iterations", e.search.iterations]);
    facts.push(["Solver exploitability", `${(e.search.exploitability_chips / 100).toFixed(3)} bb`]);
  }
  const trans = [...new Set((e.abstract_histories || []).flatMap((h) => h.translations))];
  block.innerHTML =
    `<div class="dline"><span>${d.street} · holding ${d.hole.join(" ")}</span>` +
    `<span class="source">${d.source === "subgame_solve" ? "solved live" : "blueprint"}</span></div>` +
    `<div class="bars">${bars}</div>` +
    `<dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
    (trans.length ? `<div class="trans">Mapped bets: ${trans.join("; ")}</div>` : "");
  return block;
}

function renderDecisions(s) {
  const list = $("decisions");
  list.innerHTML = "";
  [...s.decisions].reverse().forEach((d) => list.appendChild(decisionBlock(d)));
  const empty = $("decision-empty");
  empty.hidden = s.decisions.length > 0;
  empty.textContent = s.decisions_hidden
    ? "Regret has acted. Its reasoning (and its cards) are shown when the hand is over."
    : "Regret's decisions appear here: what it could do, how often it does each, and the numbers behind it.";
}

function renderLog(log) {
  const el = $("log");
  el.innerHTML = log.map((l) => `<li><span class="st">${l.street}</span>${l.who} ${l.text}</li>`).join("");
  el.scrollTop = el.scrollHeight;
}

function renderResult(s) {
  const el = $("result");
  el.hidden = !s.result;
  if (!s.result) return;
  const agentWon = s.result.payoffs_bb[s.agent_seat];
  const subject = s.mode === "play" ? (agentWon < 0 ? "You win" : agentWon > 0 ? "Regret wins" : "Split pot")
    : (agentWon > 0 ? "Regret wins" : agentWon < 0 ? `${s.seats[1 - s.agent_seat].name} wins` : "Split pot");
  el.innerHTML = `<span><b>${subject}</b> ${Math.abs(agentWon)} bb${s.result.showdown ? " at showdown" : ""}</span>`;
  const next = document.createElement("button");
  next.className = "primary";
  next.textContent = "Next hand";
  next.addEventListener("click", () => run(() => api(`sessions/${session.id}/next`)));
  el.appendChild(next);
}

function act(kind, amount) {
  run(() => api(`sessions/${session.id}/act`, { kind, amount_bb: amount ?? null }));
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
  run(() => api("sessions", { mode, opponent: $("opponent").value }));
}

$("mode-play").addEventListener("click", () => start("play"));
$("mode-watch").addEventListener("click", () => start("watch"));
$("opponent").addEventListener("change", () => start("watch"));
$("step").addEventListener("click", () => run(() => api(`sessions/${session.id}/step`)));
$("auto").addEventListener("click", () => {
  if (autoTimer) return stopAuto();
  $("auto").textContent = "Pause";
  autoTimer = setInterval(() => {
    const path = session.result ? "next" : "step";
    run(() => api(`sessions/${session.id}/${path}`));
  }, 900);
});

start("play");
