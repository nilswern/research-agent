/* ResearchPilot UI: SSE-Client + kleiner Markdown-Renderer (ohne Abhängigkeiten). */

const $ = (id) => document.getElementById(id);

const el = {
  form: $("ask"), question: $("question"), run: $("run"), stop: $("stop"),
  maxSteps: $("maxSteps"), results: $("results"), repairs: $("repairs"),
  language: $("language"), save: $("save"),
  planCard: $("planCard"), plan: $("plan"), log: $("log"),
  reports: $("reports"), chips: $("chips"), dot: $("statusDot"),
  report: $("report"), reportTitle: $("reportTitle"), meta: $("meta"),
  issues: $("issues"), copy: $("copy"),
  sourcesCard: $("sourcesCard"), sources: $("sources"), sourceCount: $("sourceCount"),
  purge: $("purge"), quit: $("quit"), overlay: $("overlay"),
};

let controller = null;
let currentMarkdown = "";

/* --------------------------- Markdown --------------------------- */

const escapeHtml = (text) =>
  text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function inlineMarkdown(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(
    /\[([^\]]*)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  out = out.replace(
    /(^|[\s(])(https?:\/\/[^\s<)"]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
  );
  return out;
}

function renderTable(rows) {
  const cells = (line) =>
    line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
  const [head, , ...body] = rows;
  const headHtml = cells(head).map((c) => `<th>${inlineMarkdown(c)}</th>`).join("");
  const bodyHtml = body
    .map((row) => `<tr>${cells(row).map((c) => `<td>${inlineMarkdown(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];

  const flush = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flush();
      const code = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (/^\s*$/.test(line)) { flush(); continue; }

    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { flush(); html.push("<hr />"); continue; }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flush();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      flush();
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      i -= 1;
      html.push(renderTable(rows));
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flush();
      const quote = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      i -= 1;
      html.push(`<blockquote>${inlineMarkdown(quote.join(" "))}</blockquote>`);
      continue;
    }

    const bullet = /^\s*([-*+])\s+/;
    const numbered = /^\s*\d+[.)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      flush();
      const ordered = numbered.test(line);
      const pattern = ordered ? numbered : bullet;
      const items = [];
      while (i < lines.length && pattern.test(lines[i])) items.push(lines[i++].replace(pattern, ""));
      i -= 1;
      const tag = ordered ? "ol" : "ul";
      html.push(`<${tag}>${items.map((it) => `<li>${inlineMarkdown(it)}</li>`).join("")}</${tag}>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  flush();
  return html.join("\n");
}

/* --------------------------- UI-Helfer --------------------------- */

function setStatus(state) {
  el.dot.className = `dot ${state}`;
}

function addLog(text, level = "info") {
  if (el.log.firstElementChild?.classList.contains("muted")) el.log.innerHTML = "";
  const item = document.createElement("li");
  item.className = level;
  item.textContent = text;
  el.log.appendChild(item);
  el.log.scrollTop = el.log.scrollHeight;
}

function showIssues(issues, isError = false) {
  if (!issues.length) { el.issues.hidden = true; return; }
  el.issues.hidden = false;
  el.issues.className = `issues${isError ? " error" : ""}`;
  const title = isError ? "Fehler" : "Offene Quellen-Probleme";
  el.issues.innerHTML =
    `<b>${title}</b><ul>${issues.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`;
}

function showSources(sources) {
  if (!sources.length) { el.sourcesCard.hidden = true; return; }
  el.sourcesCard.hidden = false;
  el.sourceCount.textContent = String(sources.length);
  el.sources.innerHTML = sources
    .map((url) => `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a></li>`)
    .join("");
}

function showReport(markdown) {
  currentMarkdown = markdown;
  el.report.innerHTML = renderMarkdown(markdown);
  el.copy.hidden = !markdown;
}

function setRunning(running) {
  el.run.disabled = running;
  el.run.textContent = running ? "Recherche läuft …" : "Recherche starten";
  el.stop.hidden = !running;
}

/* --------------------------- Backend --------------------------- */

async function loadStats() {
  const stats = await fetch("/api/stats").then((r) => r.json());
  el.chips.innerHTML = [
    `<span class="chip"><b>${stats.chunks}</b> Chunks</span>`,
    `<span class="chip"><b>${stats.documents}</b> Dokumente</span>`,
    `<span class="chip">${escapeHtml(stats.llm_model)}</span>`,
  ].join("");
  if (!el.maxSteps.value) el.maxSteps.value = stats.defaults.max_steps;
  if (!el.results.value) el.results.value = stats.defaults.results;
  if (!el.repairs.value) el.repairs.value = stats.defaults.repairs;
  el.language.value = stats.defaults.language;
}

async function loadReports() {
  const { reports } = await fetch("/api/reports").then((r) => r.json());
  if (!reports.length) {
    el.reports.innerHTML = '<li class="muted">Noch keine gespeicherten Reports.</li>';
    return;
  }
  el.reports.innerHTML = reports
    .map(
      (r) =>
        `<li><button type="button" data-name="${escapeHtml(r.name)}">${escapeHtml(r.question)}` +
        `<small>${new Date(r.modified * 1000).toLocaleString()}</small></button></li>`
    )
    .join("");
}

async function openReport(name) {
  const data = await fetch(`/api/reports/${encodeURIComponent(name)}`).then((r) => r.json());
  const body = data.content.replace(/^---\n[\s\S]*?\n---\n/, "");
  const sources = [...data.content.matchAll(/^ {2}- (https?:\/\/\S+)$/gm)].map((m) => m[1]);
  el.reportTitle.textContent = "Gespeicherter Report";
  el.meta.textContent = name;
  showIssues([]);
  showReport(body.trim());
  showSources(sources);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function startResearch(question) {
  controller = new AbortController();
  setRunning(true);
  setStatus("busy");
  el.log.innerHTML = "";
  el.planCard.hidden = true;
  el.plan.textContent = "";
  showIssues([]);
  showSources([]);
  el.meta.textContent = "";
  el.reportTitle.textContent = "Report";
  el.copy.hidden = true;
  el.report.innerHTML = '<pre class="stream" id="streamOut"></pre>';
  const streamOut = $("streamOut");

  const payload = {
    question,
    max_steps: Number(el.maxSteps.value) || null,
    results: Number(el.results.value) || null,
    repairs: el.repairs.value === "" ? null : Number(el.repairs.value),
    language: el.language.value,
    save: el.save.checked,
  };

  let response;
  try {
    response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    finish("error", `Verbindung fehlgeschlagen: ${error.message}`);
    return;
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    finish("error", detail.detail || "Unbekannter Fehler");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (line) handleEvent(JSON.parse(line.slice(6)), streamOut);
      }
    }
    if (el.run.disabled) finish("done");
  } catch (error) {
    if (error.name === "AbortError") finish("done", null, "Abgebrochen.");
    else finish("error", error.message);
  }
}

function handleEvent(event, streamOut) {
  if (event.type === "status") {
    addLog(event.text, event.level);
  } else if (event.type === "plan") {
    el.planCard.hidden = false;
    el.plan.textContent = event.text;
  } else if (event.type === "token") {
    streamOut.textContent += event.text;
  } else if (event.type === "report") {
    streamOut.textContent = event.text;
  } else if (event.type === "done") {
    showReport(event.report);
    showSources(event.sources);
    showIssues(event.issues);
    el.meta.textContent =
      `${event.research_steps} Runden · ${event.sources.length} Quellen · ` +
      `${event.repair_attempts} Repairs${event.saved ? ` · ${event.saved}` : ""}`;
    addLog(event.saved ? `Gespeichert: ${event.saved}` : "Fertig (nicht gespeichert)", "ok");
    finish("done");
    loadStats();
    loadReports();
  } else if (event.type === "error") {
    showIssues([event.text], true);
    finish("error", event.text);
  }
}

function finish(state, errorText, note) {
  setRunning(false);
  setStatus(state);
  controller = null;
  if (errorText) addLog(errorText, "error");
  if (note) addLog(note, "warn");
  const stream = $("streamOut");
  if (stream) stream.classList.remove("stream");
}

/* --------------------------- Events --------------------------- */

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (!question || el.run.disabled) return;
  startResearch(question);
});

el.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) el.form.requestSubmit();
});

el.stop.addEventListener("click", () => controller?.abort());

el.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(currentMarkdown);
  el.copy.textContent = "Kopiert ✓";
  setTimeout(() => { el.copy.textContent = "Markdown kopieren"; }, 1500);
});

el.reports.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-name]");
  if (button) openReport(button.dataset.name);
});

el.purge.addEventListener("click", async () => {
  el.purge.disabled = true;
  try {
    const result = await fetch("/api/purge", { method: "POST" }).then((r) => r.json());
    addLog(`${result.removed} veraltete Chunks entfernt`, "ok");
    await loadStats();
  } finally {
    el.purge.disabled = false;
  }
});

el.quit.addEventListener("click", async () => {
  const running = el.run.disabled;
  const question = running
    ? "Es läuft gerade eine Recherche. Trotzdem beenden?"
    : "ResearchPilot beenden?";
  if (!confirm(question)) return;
  controller?.abort();
  try {
    await fetch("/api/shutdown", { method: "POST" });
  } catch {
    /* Server ist beim Beenden weg - das ist der Normalfall. */
  }
  el.overlay.hidden = false;
});

loadStats();
loadReports();
