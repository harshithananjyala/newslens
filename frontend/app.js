"use strict";

const $ = (sel) => document.querySelector(sel);
const els = {
  form: $("#search-form"), q: $("#q"), topic: $("#topic"), status: $("#status"),
  out: $("#out"), help: $("#mode-help"), stats: $("#stats"), results: $("#results"),
};

const MODES = {
  hybrid: { label: "Hybrid", help: "Blends keyword matches and meaning matches. Best overall." },
  semantic: { label: "Semantic", help: "Matches meaning, even when the articles use different words." },
  keyword: { label: "Keyword", help: "Matches the exact words you typed, ranked with BM25." },
  compare: { label: "Compare all", help: "See what each method finds side by side. Tags show which results the others missed." },
};
const METHODS = ["keyword", "semantic", "hybrid"];
const SUBTITLE = {
  keyword: "Needs your exact words",
  semantic: "Understands meaning",
  hybrid: "Fuses both lists",
};

let controller = null;
let debounceTimer = null;
let similarTo = null;

/* ---------- helpers ---------- */
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function highlight(text, terms) {
  if (!terms || !terms.length) return escapeHtml(text);
  const re = new RegExp(`\\b(${terms.map(escapeRegex).join("|")})\\b`, "gi");
  let out = "", last = 0, m;
  while ((m = re.exec(text))) {
    out += escapeHtml(text.slice(last, m.index)) + "<mark>" + escapeHtml(m[0]) + "</mark>";
    last = m.index + m[0].length;
  }
  return out + escapeHtml(text.slice(last));
}

const currentMode = () => document.querySelector("input[name=mode]:checked").value;
const setStatus = (msg) => { els.status.textContent = msg; };

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => { if (v !== "" && v != null) url.searchParams.set(k, v); });
  if (controller) controller.abort();
  controller = new AbortController();
  const res = await fetch(url, { signal: controller.signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res.json();
}

/* ---------- rendering ---------- */
function scoreLabel(method, score) {
  if (method === "semantic") return `Similarity ${score.toFixed(2)}`;
  if (method === "keyword") return `Keyword score ${score.toFixed(1)}`;
  return `Blend score ${(score * 100).toFixed(1)}`;
}

function pills(item, method, context) {
  const out = [];
  if (context && context.sets) {
    const others = METHODS.filter((m) => m !== method && context.sets[m].has(item.id));
    out.push(others.length
      ? `<span class="pill">Also in ${others.map((m) => MODES[m].label.toLowerCase()).join(" and ")}</span>`
      : `<span class="pill only">Only in this list</span>`);
  } else if (method === "hybrid") {
    const k = item.ranks.keyword, s = item.ranks.semantic;
    out.push(`<span class="pill">${k ? `Keyword rank ${k}` : "Not in keyword results"}</span>`);
    out.push(`<span class="pill">${s ? `Semantic rank ${s}` : "Not in semantic results"}</span>`);
  }
  return out.join("");
}

function card(item, index, terms, method, context, showSimilar = true) {
  return `
    <li class="card" data-method="${method}">
      <div class="rank" aria-hidden="true">${index + 1}</div>
      <div>
        <p class="text">${highlight(item.text, terms)}</p>
        <div class="meta">
          <span class="tag">${escapeHtml(item.category)}</span>
          <span>${scoreLabel(method, item.score)}</span>
          ${pills(item, method, context)}
          ${showSimilar ? `<button type="button" class="more" data-id="${item.id}">More like this</button>` : ""}
        </div>
      </div>
    </li>`;
}

function renderList(data, method) {
  const n = data.results.length;
  if (!n) {
    setStatus("No matches");
    els.out.innerHTML = `<div class="empty"><strong>No articles matched.</strong> Try different words, or set the topic to All topics.</div>`;
    return;
  }
  setStatus(`${n} results in ${data.took_ms} ms`);
  els.out.innerHTML = `<ol class="list">${data.results.map((r, i) => card(r, i, data.terms, method, null)).join("")}</ol>`;
}

function renderSimilar(data) {
  setStatus(`${data.results.length} similar articles in ${data.took_ms} ms`);
  els.out.innerHTML = `
    <div class="banner">
      <p><strong>Articles similar to:</strong> ${escapeHtml(data.source.text)}</p>
      <button type="button" id="clear-similar">Back to search</button>
    </div>
    <ol class="list">${data.results.map((r, i) => card(r, i, [], "semantic", null)).join("")}</ol>`;
}

function renderCompare(data) {
  const sets = {};
  METHODS.forEach((m) => { sets[m] = new Set(data[m].results.map((r) => r.id)); });
  const total = Math.max(data.keyword.results.length, data.semantic.results.length);
  setStatus(`Compared in ${data.took_ms} ms`);
  const insight = total
    ? `Keyword and semantic search agree on ${data.overlap} of ${total} results. The rest were found by only one method.`
    : "";
  const cols = METHODS.map((m) => `
    <div class="col" data-method="${m}">
      <h2>${MODES[m].label}</h2>
      <p class="sub">${SUBTITLE[m]}</p>
      ${data[m].results.length
        ? `<ol class="list">${data[m].results.map((r, i) => card(r, i, data.terms, m, { sets }, false)).join("")}</ol>`
        : `<div class="empty">No matches from this method.</div>`}
    </div>`).join("");
  els.out.innerHTML = `${insight ? `<p class="insight">${insight}</p>` : ""}<div class="cols">${cols}</div>`;
}

function showIntro() {
  setStatus("");
  els.out.innerHTML = `<div class="empty">Start with an example above, or type your own question. Switch to <strong>Compare all</strong> to see how keyword and semantic search differ.</div>`;
}

function showError(err) {
  setStatus("Something went wrong");
  const offline = err instanceof TypeError;
  els.out.innerHTML = `<div class="error"><strong>${offline ? "Can't reach the search server." : "Search failed."}</strong> ${
    offline ? "Start it with <code>./run.sh</code> and reload this page." : escapeHtml(err.message)}</div>`;
}

/* ---------- actions ---------- */
function syncUrl() {
  const params = new URLSearchParams();
  if (els.q.value.trim()) params.set("q", els.q.value.trim());
  if (currentMode() !== "hybrid") params.set("mode", currentMode());
  if (els.topic.value) params.set("topic", els.topic.value);
  const qs = params.toString();
  history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
}

async function run() {
  similarTo = null;
  const q = els.q.value.trim();
  const mode = currentMode();
  els.help.textContent = MODES[mode].help;
  syncUrl();
  if (q.length < 2) { showIntro(); return; }
  els.out.setAttribute("aria-busy", "true");
  setStatus("Searching");
  try {
    if (mode === "compare") renderCompare(await api("/api/compare", { q, category: els.topic.value }));
    else renderList(await api("/api/search", { q, mode, category: els.topic.value }), mode);
  } catch (err) {
    if (err.name === "AbortError") return;
    showError(err);
  } finally {
    els.out.removeAttribute("aria-busy");
  }
}

async function findSimilar(id) {
  els.out.setAttribute("aria-busy", "true");
  setStatus("Finding similar articles");
  try {
    similarTo = id;
    renderSimilar(await api(`/api/similar/${id}`));
    els.results.focus({ preventScroll: true });
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    if (err.name !== "AbortError") showError(err);
  } finally {
    els.out.removeAttribute("aria-busy");
  }
}

/* ---------- events ---------- */
els.form.addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(debounceTimer); run(); });
els.q.addEventListener("input", () => { clearTimeout(debounceTimer); debounceTimer = setTimeout(run, 350); });
document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", run));
els.topic.addEventListener("change", run);

$("#examples").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  els.q.value = e.target.textContent;
  run();
});

els.out.addEventListener("click", (e) => {
  const more = e.target.closest(".more");
  if (more) return findSimilar(Number(more.dataset.id));
  if (e.target.id === "clear-similar") run();
});

document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (e.key === "/" && !typing) { e.preventDefault(); els.q.focus(); els.q.select(); }
});

/* ---------- start ---------- */
(async function init() {
  const params = new URLSearchParams(window.location.search);
  els.q.value = params.get("q") || "";
  if (params.get("topic")) els.topic.value = params.get("topic");
  const mode = params.get("mode");
  if (mode && MODES[mode]) document.querySelector(`input[name=mode][value=${mode}]`).checked = true;
  els.help.textContent = MODES[currentMode()].help;

  try {
    const s = await api("/api/stats");
    els.stats.textContent = `${s.articles.toLocaleString()} articles indexed with ${s.model}`;
  } catch (err) { /* shown when a search runs */ }

  if (els.q.value.trim().length >= 2) run(); else showIntro();
})();
