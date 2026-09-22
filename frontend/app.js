"use strict";

const STORAGE_KEY = "financial-agent-current-session-v2";
const POLL_INTERVAL_MS = 700;

const el = Object.fromEntries([
  "connectionDot", "connectionLabel", "modelLabel", "sessionTitle", "turnCount",
  "currentRunId", "sessionStatus", "clearConversation", "conversationRunState",
  "messageStream", "composerForm", "queryInput", "historyHint", "sendButton",
  "rawTraceButton", "runSummary", "runTimeline", "rawTraceDialog",
  "closeRawTrace", "rawTraceContent", "rawTraceMeta", "copyRawTrace", "toast",
].map((id) => [id, document.getElementById(id)]));

let state = loadState();
let latestSnapshot = null;
let activeRunId = null;
let toastTimer = null;

function t(value) {
  const labels = {
    newConversation: "\u65b0\u4f1a\u8bdd", waiting: "\u7b49\u5f85\u8f93\u5165", idle: "\u5f85\u547d",
    running: "\u8fd0\u884c\u4e2d", completed: "\u5df2\u5b8c\u6210", failed: "\u8fd0\u884c\u5931\u8d25",
    queued: "\u6392\u961f\u4e2d", you: "\u4f60", agent: "Agent", retry: "\u91cd\u8bd5",
    copy: "\u590d\u5236", copied: "\u5df2\u590d\u5236", copyFailed: "\u590d\u5236\u5931\u8d25",
  };
  return labels[value] || value;
}

function loadState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (parsed && Array.isArray(parsed.messages)) {
      return { messages: parsed.messages, lastRunId: parsed.lastRunId || null };
    }
  } catch (_) { /* Ignore broken local state. */ }
  return { messages: [], lastRunId: null };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function inlineMarkdown(value) {
  const code = [];
  let text = escapeHtml(value).replace(/`([^`\n]+)`/g, (_, body) => {
    const token = `@@CODE${code.length}@@`;
    code.push(`<code>${body}</code>`);
    return token;
  });
  text = text
    .replace(/\*\*([^*\n][\s\S]*?)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n][\s\S]*?)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  code.forEach((item, index) => { text = text.replace(`@@CODE${index}@@`, item); });
  return text;
}

function isTableDivider(line) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function renderMarkdown(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    if (/^```/.test(line.trim())) {
      const language = line.trim().slice(3).trim();
      const body = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i].trim())) body.push(lines[i++]);
      if (i < lines.length) i += 1;
      output.push(`<pre><code${language ? ` data-language="${escapeHtml(language)}"` : ""}>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      i += 1; continue;
    }
    if (i + 1 < lines.length && line.includes("|") && isTableDivider(lines[i + 1])) {
      const headers = tableCells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(tableCells(lines[i++]));
      output.push(`<table><thead><tr>${headers.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const listMatch = line.match(/^\s*([-*+] |\d+\. )(.+)$/);
    if (listMatch) {
      const ordered = /\d/.test(listMatch[1][0]);
      const tag = ordered ? "ol" : "ul";
      const items = [];
      while (i < lines.length) {
        const match = lines[i].match(/^\s*([-*+] |\d+\. )(.+)$/);
        if (!match || /\d/.test(match[1][0]) !== ordered) break;
        items.push(`<li>${inlineMarkdown(match[2])}</li>`); i += 1;
      }
      output.push(`<${tag}>${items.join("")}</${tag}>`); continue;
    }
    if (/^>\s?/.test(line)) {
      const quotes = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quotes.push(lines[i++].replace(/^>\s?/, ""));
      output.push(`<blockquote>${quotes.map(inlineMarkdown).join("<br>")}</blockquote>`); continue;
    }
    const paragraph = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4})\s+/.test(lines[i]) && !/^```/.test(lines[i].trim()) && !/^\s*([-*+] |\d+\. )/.test(lines[i]) && !/^>\s?/.test(lines[i])) {
      if (i + 1 < lines.length && lines[i].includes("|") && isTableDivider(lines[i + 1])) break;
      paragraph.push(lines[i++]);
    }
    output.push(`<p>${paragraph.map(inlineMarkdown).join("<br>")}</p>`);
  }
  return output.join("");
}

function formatTime(iso) {
  if (!iso) return "";
  try { return new Date(iso).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }); }
  catch (_) { return ""; }
}

function messageNode(message, index) {
  const article = document.createElement("article");
  article.className = `message ${message.role}${message.failed ? " failed" : ""}`;
  const head = document.createElement("div");
  head.className = "message-head";
  const name = document.createElement("strong");
  name.textContent = message.role === "user" ? t("you") : t("agent");
  const time = document.createElement("span");
  time.textContent = formatTime(message.createdAt);
  head.append(name, time);
  const body = document.createElement("div");
  body.className = "message-body";
  if (message.role === "assistant" && !message.failed) {
    body.classList.add("markdown");
    body.innerHTML = renderMarkdown(message.content);
  } else {
    body.textContent = message.content;
  }
  article.append(head, body);
  if (message.failed && message.query) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const retry = document.createElement("button");
    retry.type = "button"; retry.className = "retry-button"; retry.dataset.retryIndex = String(index);
    retry.innerHTML = `<svg><use href="#icon-refresh"></use></svg>${t("retry")}`;
    actions.append(retry); article.append(actions);
  }
  return article;
}

function renderMessages() {
  el.messageStream.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-conversation";
    empty.innerHTML = "<strong>Financial Agent</strong><p>&#21487;&#20197;&#35810;&#38382;&#25345;&#20179;&#39118;&#38505;&#12289;&#24066;&#22330;&#25968;&#25454;&#25110;&#20844;&#21578;&#24433;&#21709;&#12290;&#21491;&#20391;&#20250;&#21516;&#27493;&#26174;&#31034; Agent &#30340;&#23436;&#25972;&#25191;&#34892;&#36807;&#31243;&#12290;</p>";
    el.messageStream.append(empty);
  } else {
    state.messages.forEach((message, index) => el.messageStream.append(messageNode(message, index)));
  }
  el.messageStream.scrollTop = el.messageStream.scrollHeight;
  updateSessionMeta();
}

function showThinking() {
  const article = document.createElement("article");
  article.id = "thinkingMessage"; article.className = "message assistant thinking-message";
  article.innerHTML = `<div class="message-head"><strong>Agent</strong></div><div class="message-body"><span>Agent \u6b63\u5728\u8fd0\u884c</span><span class="thinking-dots"></span></div>`;
  el.messageStream.append(article); el.messageStream.scrollTop = el.messageStream.scrollHeight;
}

function updateSessionMeta() {
  const first = state.messages.find((message) => message.role === "user");
  el.sessionTitle.textContent = first ? first.content.replace(/\s+/g, " ").slice(0, 24) : t("newConversation");
  el.turnCount.textContent = String(state.messages.filter((message) => message.role === "user").length);
  el.currentRunId.textContent = state.lastRunId ? state.lastRunId.slice(0, 8) : "\u2014";
  el.currentRunId.title = state.lastRunId || "";
  el.historyHint.textContent = `\u5c06\u643a\u5e26 ${state.messages.filter((message) => !message.failed).length} \u6761\u5386\u53f2\u6d88\u606f`;
}

function setBusy(isBusy, label) {
  el.queryInput.disabled = isBusy; el.sendButton.disabled = isBusy; el.clearConversation.disabled = isBusy;
  el.sessionStatus.textContent = label || (isBusy ? t("running") : t("idle"));
  el.conversationRunState.textContent = label || (isBusy ? t("running") : t("waiting"));
  el.conversationRunState.classList.toggle("running", isBusy);
}

function historyPayload(messages = state.messages) {
  return messages.filter((item) => !item.failed && (item.role === "user" || item.role === "assistant"))
    .map(({ role, content }) => ({ role, content }));
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(body?.detail || `${response.status} ${response.statusText}`);
  return body;
}

async function startRun(query, history) {
  return api("/api/runs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, capture_mode: "evaluation" }),
  });
}

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

async function pollRun(runId) {
  while (activeRunId === runId) {
    const snapshot = await api(`/api/runs/${runId}`);
    latestSnapshot = snapshot; renderRun(snapshot);
    if (snapshot.status === "completed" || snapshot.status === "failed") return snapshot;
    await sleep(POLL_INTERVAL_MS);
  }
  throw new Error("Run cancelled");
}

function errorText(snapshot, fallback) {
  if (!snapshot) return fallback;
  const code = snapshot.error?.code || "AGENT_RUN_FAILED";
  const type = snapshot.error?.type || "UnknownError";
  const detail = snapshot.diagnostics?.message || snapshot.diagnostics?.detail;
  return `${code} \u00b7 ${type}${detail ? `\n${detail}` : ""}`;
}

async function runQuery(query, history, appendUser = true) {
  if (activeRunId) return;
  if (appendUser) state.messages.push({ role: "user", content: query, createdAt: new Date().toISOString() });
  saveState(); renderMessages(); showThinking(); setBusy(true, t("queued"));
  try {
    const started = await startRun(query, history);
    activeRunId = started.run_id; state.lastRunId = started.run_id; saveState(); updateSessionMeta();
    const snapshot = await pollRun(started.run_id);
    document.getElementById("thinkingMessage")?.remove();
    if (snapshot.status === "completed" && snapshot.result?.answer) {
      state.messages.push({ role: "assistant", content: snapshot.result.answer, createdAt: new Date().toISOString(), runId: started.run_id });
      setBusy(false, t("completed"));
    } else if (snapshot.status === "completed") {
      const noAnswer = snapshot.result?.status === "clarify" ? "Agent \u9700\u8981\u66f4\u591a\u4fe1\u606f\u624d\u80fd\u7ee7\u7eed\u3002" : "Agent \u5df2\u7ed3\u675f\u8fd0\u884c\uff0c\u4f46\u6ca1\u6709\u751f\u6210\u56de\u7b54\u3002";
      state.messages.push({ role: "assistant", content: noAnswer, createdAt: new Date().toISOString(), runId: started.run_id });
      setBusy(false, t("completed"));
    } else {
      state.messages.push({ role: "assistant", content: errorText(snapshot, t("failed")), failed: true, query, createdAt: new Date().toISOString(), runId: started.run_id });
      setBusy(false, t("failed"));
    }
  } catch (error) {
    document.getElementById("thinkingMessage")?.remove();
    state.messages.push({ role: "assistant", content: error.message || String(error), failed: true, query, createdAt: new Date().toISOString(), runId: activeRunId });
    setBusy(false, t("failed"));
  } finally {
    activeRunId = null; saveState(); renderMessages();
  }
}

function projectionValue(projection) {
  if (!projection) return null;
  return Object.prototype.hasOwnProperty.call(projection, "value") ? projection.value : { capture_mode: projection.capture_mode, digest: projection.digest, item_count: projection.item_count };
}

function pretty(value) {
  if (typeof value === "string") return value;
  try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
}

function addText(parent, tag, className, text) {
  const node = document.createElement(tag); node.className = className || ""; node.textContent = text; parent.append(node); return node;
}

function addJson(body, heading, value) {
  addText(body, "h4", "", heading);
  const pre = addText(body, "pre", "json-block", pretty(value));
  pre.tabIndex = 0;
}

function addMeta(body, values) {
  const row = document.createElement("div"); row.className = "trace-meta";
  values.filter(Boolean).forEach((value) => addText(row, "span", "", value)); body.append(row);
}

function contextMessages(body, messages) {
  addText(body, "h4", "", "\u8f93\u5165\u4e0a\u4e0b\u6587");
  if (!Array.isArray(messages)) { addJson(body, "", messages); return; }
  messages.forEach((message, index) => {
    const item = document.createElement("section"); item.className = "context-message";
    addText(item, "header", "", `${index + 1} \u00b7 ${message.role || "message"}`);
    addText(item, "pre", "", typeof message.content === "string" ? message.content : pretty(message.content));
    body.append(item);
  });
}

function eventTitle(event) {
  const names = {
    run_started: "\u5f00\u59cb\u8fd0\u884c", context_selected: "\u9009\u62e9\u4e0a\u4e0b\u6587",
    plan_proposed: "Planner \u8f93\u51fa\u8ba1\u5212", plan_validated: "\u8ba1\u5212\u7ed3\u6784\u6821\u9a8c",
    tool_attempt: "Tool \u8c03\u7528", tool_reuse: "\u590d\u7528 Tool \u7ed3\u679c",
    tool_attempt_blocked: "Tool \u8c03\u7528\u88ab\u963b\u6b62", retry_scheduled: "Tool \u5c06\u91cd\u8bd5",
    retry_skipped: "Tool \u8df3\u8fc7\u91cd\u8bd5", writer_completed: "Writer \u751f\u6210\u56de\u7b54",
    verifier_completed: "Verifier \u5224\u65ad", loop_iteration_completed: "\u672c\u8f6e\u7ed3\u675f",
    component_failed: "\u7ec4\u4ef6\u5931\u8d25", run_finished: "\u8fd0\u884c\u5b8c\u6210", run_failed: "\u8fd0\u884c\u5931\u8d25",
    trace_degraded: "\u8ffd\u8e2a\u964d\u7ea7",
  };
  if (event.kind === "model_call_started") return `${capitalize(event.component)} \u6a21\u578b\u8c03\u7528 #${event.call_index}`;
  if (event.kind === "tool_attempt") return `${event.tool_name} \u00b7 \u5c1d\u8bd5 ${event.attempt_index}`;
  if (event.kind === "context_selected") return `${capitalize(event.component)} \u4e0a\u4e0b\u6587\u9009\u62e9`;
  return names[event.kind] || event.kind;
}

function capitalize(value) { return value ? value[0].toUpperCase() + value.slice(1) : "Model"; }

function eventType(event) {
  if (event.component) return event.component;
  if (event.kind.startsWith("plan")) return "planner";
  if (event.kind.startsWith("tool") || event.kind.startsWith("retry")) return "tool";
  if (event.kind.startsWith("writer")) return "writer";
  if (event.kind.startsWith("verifier")) return "verifier";
  return "system";
}

function traceCard(event, paired) {
  const card = document.createElement("details"); card.className = "trace-card";
  if (["plan_proposed", "verifier_completed", "component_failed", "run_failed"].includes(event.kind)) card.open = true;
  const summary = document.createElement("summary");
  addText(summary, "span", `trace-type ${eventType(event)}`, eventType(event).toUpperCase());
  addText(summary, "span", "trace-title", eventTitle(event));
  addText(summary, "span", "trace-time", `${Math.round(event.elapsed_ms || 0)} ms`);
  const body = document.createElement("div"); body.className = "trace-body";
  addMeta(body, [`seq ${event.sequence}`, event.plan_revision != null ? `plan r${event.plan_revision}` : null]);

  if (event.kind === "run_started") {
    addJson(body, "\u95ee\u9898", event.query || event.query_ref); addJson(body, "\u5386\u53f2\u6d88\u606f", event.history || []);
  } else if (event.kind === "context_selected") {
    addJson(body, "\u9009\u62e9\u6307\u6807", event.metrics); addJson(body, "\u9009\u4e2d\u5185\u5bb9", event.selected_content || []); addJson(body, "\u68c0\u7d22\u5185\u5bb9", event.retrieved_content || []);
  } else if (event.kind === "model_call_started") {
    addMeta(body, [event.model, `call #${event.call_index}`]);
    contextMessages(body, projectionValue(event.messages));
    addJson(body, "\u671f\u671b\u54cd\u5e94 Schema", projectionValue(event.response_schema));
    if (paired?.completed) addJson(body, "\u6a21\u578b\u539f\u59cb\u8f93\u51fa", projectionValue(paired.completed.response));
    if (paired?.failed) addJson(body, "\u8c03\u7528\u5931\u8d25", paired.failed);
    if (!paired?.completed && !paired?.failed) addText(body, "p", "trace-note", "\u7b49\u5f85\u6a21\u578b\u8fd4\u56de\u2026");
  } else if (event.kind === "plan_proposed") {
    addMeta(body, [`decision ${event.decision}`, `${event.tasks?.length || 0} tasks`]); addJson(body, "Planner \u8f93\u51fa", { decision: event.decision, tasks: event.tasks, force_rerun_task_ids: event.force_rerun_task_ids || event.force_rerun_refs });
  } else if (event.kind === "plan_validated") {
    addMeta(body, [event.valid ? "valid" : "invalid", event.decision]); addJson(body, "\u6821\u9a8c\u7ed3\u679c", { issue_codes: event.issue_codes, task_ids: event.task_ids || event.task_refs });
  } else if (event.kind === "tool_attempt") {
    addMeta(body, [event.result_status, `${Math.round(event.latency_ms)} ms`, event.retryable ? "retryable" : null]);
    addJson(body, "\u8c03\u7528\u53c2\u6570", projectionValue(event.arguments)); addJson(body, "Tool \u8fd4\u56de", projectionValue(event.result));
    if (event.error_code || event.exception_type) addJson(body, "\u9519\u8bef", { error_code: event.error_code, exception_type: event.exception_type });
    addJson(body, "\u9884\u7b97", { before: event.budget_before, after: event.budget_after });
  } else if (event.kind === "writer_completed") {
    addMeta(body, [event.mode, `${event.evidence?.length || 0} evidence`]); addJson(body, "\u56de\u7b54", event.answer || event.answer_ref); addJson(body, "\u8bc1\u636e\u5f15\u7528", event.evidence || []);
  } else if (event.kind === "verifier_completed") {
    const badge = addText(body, "span", `decision ${event.decision === "PASS" ? "pass" : "fail"}`, event.decision); badge.style.display = "inline-block";
    addJson(body, "Verifier \u5224\u65ad", { decision: event.decision, reason: event.reason || event.reason_ref, missing_evidence: event.missing_evidence || event.missing_evidence_refs, failed_task_ids: event.failed_task_ids || event.failed_task_refs });
  } else if (event.kind === "loop_iteration_completed") {
    addMeta(body, [event.action, event.decision, `rewrite ${event.rewrite_count}`, `replan ${event.replan_count}`]); addJson(body, "\u9884\u7b97", event.budget);
  } else if (event.kind === "run_finished") {
    addMeta(body, [event.status, event.stop_reason, `${event.iteration_count} iterations`]); addJson(body, "\u7edf\u8ba1", event);
  } else {
    addJson(body, "\u4e8b\u4ef6\u5185\u5bb9", event);
  }
  card.append(summary, body); return card;
}

function renderRun(snapshot) {
  const events = snapshot.events || snapshot.trace?.events || [];
  el.rawTraceButton.disabled = !events.length; el.rawTraceContent.textContent = pretty(snapshot.trace || snapshot);
  el.rawTraceMeta.textContent = `${events.length} events \u00b7 ${snapshot.trace_id || ""}`;
  const terminal = snapshot.status === "completed" || snapshot.status === "failed";
  const summaryClass = snapshot.status === "completed" ? "success" : snapshot.status === "failed" ? "failed" : "running";
  el.runSummary.replaceChildren();
  const dot = document.createElement("span"); dot.className = `status-indicator ${summaryClass}`;
  const info = document.createElement("div");
  addText(info, "strong", "", snapshot.status === "completed" ? "\u8fd0\u884c\u5b8c\u6210" : snapshot.status === "failed" ? "\u8fd0\u884c\u5931\u8d25" : "Agent \u6b63\u5728\u8fd0\u884c");
  const last = events.at(-1);
  addText(info, "small", "", `${events.length} \u4e2a\u4e8b\u4ef6${last ? ` \u00b7 ${Math.round(last.elapsed_ms || 0)} ms` : ""}${snapshot.result ? ` \u00b7 ${snapshot.result.iteration_count || 0} \u8f6e` : ""}`);
  el.runSummary.append(dot, info);
  if (!events.length) return;

  const pairMap = new Map();
  events.filter((event) => event.kind === "model_call_completed" || event.kind === "model_call_failed").forEach((event) => {
    const key = `${event.component}:${event.call_index}`; const pair = pairMap.get(key) || {};
    pair[event.kind === "model_call_completed" ? "completed" : "failed"] = event; pairMap.set(key, pair);
  });
  const visible = events.filter((event) => event.kind !== "model_call_completed" && event.kind !== "model_call_failed");
  const groups = new Map();
  visible.forEach((event) => {
    const iteration = event.iteration ?? 0;
    if (!groups.has(iteration)) groups.set(iteration, []);
    groups.get(iteration).push(event);
  });
  el.runTimeline.replaceChildren();
  groups.forEach((groupEvents, iteration) => {
    const section = document.createElement("section"); section.className = "iteration-group";
    const heading = document.createElement("div"); heading.className = "iteration-heading";
    heading.innerHTML = `<span>${iteration === 0 ? "SETUP" : `ITERATION ${iteration}`}</span><span>${groupEvents.length} EVENTS</span>`;
    section.append(heading);
    groupEvents.forEach((event) => {
      const pair = event.kind === "model_call_started" ? pairMap.get(`${event.component}:${event.call_index}`) : null;
      section.append(traceCard(event, pair));
    });
    el.runTimeline.append(section);
  });
  if (terminal) el.runTimeline.scrollTop = 0;
}

function showToast(message) {
  clearTimeout(toastTimer); el.toast.textContent = message; el.toast.classList.add("visible");
  toastTimer = setTimeout(() => el.toast.classList.remove("visible"), 1800);
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); showToast(t("copied")); }
  catch (_) { showToast(t("copyFailed")); }
}

async function loadEnvironment() {
  try {
    const environment = await api("/api/environment");
    el.connectionDot.className = "online"; el.connectionLabel.textContent = "\u5df2\u8fde\u63a5";
    const models = environment.models || {};
    const unique = [...new Set(Object.values(models).filter(Boolean))];
    el.modelLabel.textContent = unique.join(" / ") || "model unknown";
  } catch (_) {
    el.connectionDot.className = "offline"; el.connectionLabel.textContent = "\u672a\u8fde\u63a5"; el.modelLabel.textContent = "\u670d\u52a1\u4e0d\u53ef\u7528";
  }
}

el.composerForm.addEventListener("submit", (event) => {
  event.preventDefault(); const query = el.queryInput.value.trim(); if (!query || activeRunId) return;
  const history = historyPayload(); el.queryInput.value = ""; runQuery(query, history, true);
});

el.queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); el.composerForm.requestSubmit(); }
});

el.messageStream.addEventListener("click", (event) => {
  const button = event.target.closest("[data-retry-index]"); if (!button || activeRunId) return;
  const index = Number(button.dataset.retryIndex); const failed = state.messages[index]; if (!failed?.query) return;
  runQuery(failed.query, historyPayload(state.messages.slice(0, index)), true);
});

el.clearConversation.addEventListener("click", () => {
  if (!state.messages.length || !window.confirm("\u786e\u5b9a\u6e05\u7a7a\u5f53\u524d\u4f1a\u8bdd\uff1f")) return;
  state = { messages: [], lastRunId: null }; latestSnapshot = null; saveState(); renderMessages();
  el.rawTraceButton.disabled = true;
  el.runSummary.innerHTML = '<span class="status-indicator idle"></span><div><strong>\u5c1a\u672a\u8fd0\u884c</strong><small>\u53d1\u9001\u95ee\u9898\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u5b8c\u6574\u6267\u884c\u8fc7\u7a0b\u3002</small></div>';
  el.runTimeline.innerHTML = '<div class="empty-panel"><svg><use href="#icon-terminal"></use></svg><p>Planner\u3001Tool\u3001Writer \u548c Verifier \u7684\u8f93\u5165\u8f93\u51fa\u5c06\u6309\u8f6e\u6b21\u663e\u793a\u3002</p></div>';
});

el.rawTraceButton.addEventListener("click", () => { if (latestSnapshot) el.rawTraceDialog.showModal(); });
el.closeRawTrace.addEventListener("click", () => el.rawTraceDialog.close());
el.rawTraceDialog.addEventListener("click", (event) => { if (event.target === el.rawTraceDialog) el.rawTraceDialog.close(); });
el.copyRawTrace.addEventListener("click", () => copyText(el.rawTraceContent.textContent));

async function restoreLastRun() {
  if (!state.lastRunId) return;
  try { latestSnapshot = await api(`/api/runs/${state.lastRunId}`); renderRun(latestSnapshot); }
  catch (_) { /* A process restart drops in-memory runs; conversation remains useful. */ }
}

renderMessages(); setBusy(false); loadEnvironment(); restoreLastRun(); el.queryInput.focus();
