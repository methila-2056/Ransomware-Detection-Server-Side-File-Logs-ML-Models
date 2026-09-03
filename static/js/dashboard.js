/* ═══════════════════════════════════════════════════════════════
   Ransomware Detection System - Dashboard JS
   Clean, robust, error-free.

   Handles:
   - Socket.IO connection + real-time updates
   - Mode switching (Simulation / Real Monitor)
   - Chart.js timeline (5 operations)
   - Live operations, gauge, threat badge
   - Model comparison table (matched by model name)
   - Live + test confusion matrices
   - Runtime metrics + detection rate
   - Alert log with file details
   - Real-time event feed
   - Connection/error states
   ═══════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────
// Socket.IO + State
// ──────────────────────────────────────────────────────────────

const socket = io();

let currentMode = "idle";   // "idle" | "simulation" | "real"
let isActive = false;
let chart = null;

// Chart data series (1-second window counts)
const chartLabels = [];
const chartNC = [];
const chartNW = [];
const chartNR = [];
const chartNM = [];
const chartNU = [];
const chartAttack = [];
const MAX_POINTS = 60;

// Live detection state (for gauge/status)
let liveProb = 0;
let liveConf = 0;
let liveFamily = null;
let liveIsAttack = false;

// Model key -> display name (keys are used in both `m.name` of
// model_comparison rows AND the keys of `tick.predictions`, so the
// "Live" column is matched by key directly).
const MODEL_DISPLAY = {
    "rf": "Random Forest",
    "svm": "SVM",
    "dt": "Decision Tree",
    "ada": "AdaBoost",
    "xgb": "XGBoost",
};

// ──────────────────────────────────────────────────────────────
// Safe element helpers (avoid throwing on missing elements)
// ──────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

function setText(id, value) {
    const node = el(id);
    if (node) node.textContent = value;
}

function setNum(id, value) {
    const node = el(id);
    if (!node) return;
    const old = node.textContent;
    node.textContent = value;
    if (old !== String(value)) {
        node.classList.remove("num-pulse");
        void node.offsetWidth;
        node.classList.add("num-pulse");
    }
}

// ──────────────────────────────────────────────────────────────
// Chart.js
// ──────────────────────────────────────────────────────────────

function initChart() {
    const canvas = el("timelineChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartLabels,
            datasets: [
                { label: "Create", data: chartNC, borderColor: "#00ff88", backgroundColor: "rgba(0,255,136,0.08)", borderWidth: 1.5, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 3 },
                { label: "Write",  data: chartNW, borderColor: "#00ccff", backgroundColor: "rgba(0,204,255,0.06)", borderWidth: 1.5, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 3 },
                { label: "Read",   data: chartNR, borderColor: "#cc66ff", backgroundColor: "rgba(204,102,255,0.04)", borderWidth: 1.5, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 3 },
                { label: "Rename", data: chartNM, borderColor: "#ffcc00", backgroundColor: "rgba(255,204,0,0.04)", borderWidth: 1.5, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 3 },
                { label: "Delete", data: chartNU, borderColor: "#ff3366", backgroundColor: "rgba(255,51,102,0.04)", borderWidth: 1.5, fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 3 },
                { label: "Attack Zone", data: chartAttack, borderColor: "rgba(255,51,102,0.5)", backgroundColor: "rgba(255,51,102,0.1)", borderWidth: 1, borderDash: [4,4], fill: true, tension: 0, pointRadius: 0 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 150 },
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#111827",
                    titleColor: "#e5e7eb",
                    bodyColor: "#9ca3af",
                    borderColor: "#243044",
                    borderWidth: 1,
                    padding: 10,
                    titleFont: { family: "JetBrains Mono, monospace", size: 11 },
                    bodyFont: { family: "JetBrains Mono, monospace", size: 10 },
                    displayColors: true,
                    boxPadding: 4,
                },
            },
            scales: {
                x: { display: true, grid: { color: "rgba(36,48,68,0.3)" }, ticks: { color: "#4b5563", font: { family: "JetBrains Mono, monospace", size: 9 }, maxTicksLimit: 12 } },
                y: { display: true, beginAtZero: true, grid: { color: "rgba(36,48,68,0.3)" }, ticks: { color: "#4b5563", font: { family: "JetBrains Mono, monospace", size: 9 } } },
            },
        },
    });
}

function clearChartData() {
    chartLabels.length = 0;
    chartNC.length = 0;
    chartNW.length = 0;
    chartNR.length = 0;
    chartNM.length = 0;
    chartNU.length = 0;
    chartAttack.length = 0;
    if (chart) chart.update("none");
}

function updateChart(tick) {
    chartLabels.push((tick.timestamp || 0) + "s");
    chartNC.push(tick.nc || 0);
    chartNW.push(tick.nw || 0);
    chartNR.push(tick.nr || 0);
    chartNM.push(tick.nm || 0);
    chartNU.push(tick.nu || 0);
    chartAttack.push(tick.is_attack ? Math.max(tick.nc || 0, tick.nw || 0, tick.nr || 0, tick.nm || 0, tick.nu || 0) * 1.1 : 0);

    if (chartLabels.length > MAX_POINTS) {
        chartLabels.shift();
        chartNC.shift();
        chartNW.shift();
        chartNR.shift();
        chartNM.shift();
        chartNU.shift();
        chartAttack.shift();
    }
    if (chart) chart.update("none");
}

// ──────────────────────────────────────────────────────────────
// Live operations
// ──────────────────────────────────────────────────────────────

function updateLiveOps(tick) {
    setNum("liveNC", tick.nc || 0);
    setNum("liveNW", tick.nw || 0);
    setNum("liveNR", tick.nr || 0);
    setNum("liveNM", tick.nm || 0);
    setNum("liveNU", tick.nu || 0);

    if (el("ncBar")) setBar(el("ncBar"), (tick.nc || 0) / 200);
    if (el("nwBar")) setBar(el("nwBar"), (tick.nw || 0) / 300);
    if (el("nrBar")) setBar(el("nrBar"), (tick.nr || 0) / 400);
    if (el("nmBar")) setBar(el("nmBar"), (tick.nm || 0) / 130);
    if (el("nuBar")) setBar(el("nuBar"), (tick.nu || 0) / 130);

    setText("currentUser", tick.user || "--");
}

function setBar(barEl, ratio) {
    barEl.style.width = Math.min(100, Math.max(0, ratio * 100)) + "%";
}

// ──────────────────────────────────────────────────────────────
// Detection gauge + threat badge
// ──────────────────────────────────────────────────────────────

function updateGauge(tick) {
    const xgb = (tick.predictions && tick.predictions.xgb) || {};
    const prob = xgb.probability || 0;
    const conf = xgb.confidence || 0;
    const fam = tick.family || null;
    const isAttack = !!tick.is_attack;

    liveProb = prob;
    liveConf = conf;
    liveFamily = fam;
    liveIsAttack = isAttack;

    const arc = el("gaugeArc");
    const circumference = 314;
    if (arc) arc.setAttribute("stroke-dasharray", `${Math.floor(prob * circumference)} ${circumference}`);

    let color, levelText;
    if (prob > 0.7) { color = "#ff3366"; levelText = "CRITICAL"; }
    else if (prob > 0.4) { color = "#ffcc00"; levelText = "WARNING"; }
    else { color = "#00ff88"; levelText = "SAFE"; }

    if (arc) arc.setAttribute("stroke", color);
    if (el("gaugeValue")) { el("gaugeValue").textContent = (prob * 100).toFixed(1) + "%"; el("gaugeValue").style.color = color; }
    if (el("gaugeLabel")) { el("gaugeLabel").textContent = levelText; el("gaugeLabel").style.color = color; }

    // Threat badge
    const badge = el("threatBadge");
    const dot = el("threatDot");
    const threatLabel = el("threatLabel");
    if (badge && dot && threatLabel) {
        badge.className = "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold " +
            (prob > 0.7 ? "bg-red-500/15 text-red-400 border border-red-500/25" :
             prob > 0.4 ? "bg-yellow-500/15 text-yellow-400 border border-yellow-500/25" :
             "bg-green-500/10 text-green-400 border border-green-500/20");
        dot.className = "w-1.5 h-1.5 rounded-full bg-current " + (prob > 0.5 ? "badge-pulse" : "");
        threatLabel.textContent = prob > 0.7 ? "CRITICAL" : prob > 0.4 ? "WARNING" : "NO THREAT";
    }

    setText("confidenceVal", conf > 0 ? conf.toFixed(1) + "%" : "--");
    setText("attackFamily", fam || "None");
    setText("attackStatus", isAttack ? "ATTACK ACTIVE" : "Idle");

    // Red glow on attack
    const main = el("main");
    if (main) main.classList.toggle("glow-red", prob > 0.7);
}

// ──────────────────────────────────────────────────────────────
// Runtime metrics
// ──────────────────────────────────────────────────────────────

function updateMetrics(m) {
    if (!m) return;
    setText("metricTicks", m.total_ticks || 0);
    setText("metricAttacks", m.total_attacks || 0);
    setText("metricDetections", m.total_detections || 0);
    setText("metricFP", m.total_fp || 0);
    setText("metricFN", m.total_fn || 0);

    const rate = (m.total_attacks > 0) ? ((m.total_detections / m.total_attacks) * 100) : 0;
    setText("metricDetRate", rate.toFixed(1) + "%");
    if (el("detRateBar")) el("detRateBar").style.width = rate + "%";
}

// ──────────────────────────────────────────────────────────────
// Confusion matrices
// ──────────────────────────────────────────────────────────────

function updateLiveCM(cm) {
    if (!cm) return;
    setNum("liveCM_TP", cm.tp || 0);
    setNum("liveCM_FP", cm.fp || 0);
    setNum("liveCM_TN", Math.max(0, cm.tn || 0));
    setNum("liveCM_FN", cm.fn || 0);
}

function updateTestCM(comparison) {
    if (!comparison || comparison.length === 0) return;
    const xgb = comparison.find(m => m.name === "xgb");
    if (!xgb) return;
    setText("testCM_TP", xgb.tp || 0);
    setText("testCM_FP", xgb.fp || 0);
    setText("testCM_TN", xgb.tn || 0);
    setText("testCM_FN", xgb.fn || 0);
}

// ──────────────────────────────────────────────────────────────
// Model comparison table (matched by model name)
// ──────────────────────────────────────────────────────────────

function updateModelTable(predictions, comparison) {
    const tbody = el("modelTableBody");
    if (!tbody) return;

    if (!comparison || comparison.length === 0) {
        if (predictions) tbody.innerHTML = buildLiveOnlyRows(predictions);
        return;
    }

    // Determine best model by sensitivity
    let bestIdx = 0, bestSens = -1;
    comparison.forEach((m, i) => {
        if ((m.sensitivity || 0) > bestSens) { bestSens = m.sensitivity; bestIdx = i; }
    });

    let html = "";
    comparison.forEach((m, i) => {
        const isBest = i === bestIdx;
        const live = predictions ? predictions[m.name] : null;
        const liveText = live
            ? (live.prediction === 1 ? '<span class="text-red-400 font-bold">ATTACK</span>' : '<span class="text-green-400">SAFE</span>')
            : '<span class="text-gray-600">--</span>';

        html += `<tr class="${isBest ? 'best-row' : ''}">
            <td>${MODEL_DISPLAY[m.name] || m.name}</td>
            <td class="tabular-nums">${(m.accuracy * 100).toFixed(1)}%</td>
            <td class="tabular-nums ${(m.sensitivity || 0) > 0.9 ? 'text-neon-green' : ''}">${(m.sensitivity * 100).toFixed(1)}%</td>
            <td class="tabular-nums">${(m.f1_score * 100).toFixed(1)}%</td>
            <td class="tabular-nums text-gray-500">${m.prediction_latency_ms.toFixed(2)}ms</td>
            <td>${liveText}</td>
        </tr>`;
    });
    tbody.innerHTML = html;
    updateTestCM(comparison);
}

function buildLiveOnlyRows(predictions) {
    const names = Object.keys(MODEL_DISPLAY);
    let html = "";
    names.forEach(name => {
        const live = predictions ? predictions[name] : null;
        const liveText = live
            ? (live.prediction === 1 ? '<span class="text-red-400 font-bold">ATTACK</span>' : '<span class="text-green-400">SAFE</span>')
            : '<span class="text-gray-600">--</span>';
        html += `<tr><td>${MODEL_DISPLAY[name]}</td><td colspan="4" class="text-gray-600">--</td><td>${liveText}</td></tr>`;
    });
    return html;
}

// ──────────────────────────────────────────────────────────────
// Alerts + Event feed
// ──────────────────────────────────────────────────────────────

function ensurePlaceholderClear(container, placeholderText) {
    if (container.children.length === 1 && container.children[0].textContent.includes(placeholderText)) {
        container.innerHTML = "";
    }
}

function addAlert(alert) {
    const log = el("alertLog");
    if (!log) return;
    ensurePlaceholderClear(log, "No alerts");

    const entry = document.createElement("div");
    entry.className = "alert-enter p-2.5 rounded-xl bg-red-500/10 border border-red-500/25 text-[11px]";

    const time = new Date().toLocaleTimeString();
    const isReal = alert.source === "real";
    const source = isReal ? "REAL" : "SIM";

    let fileHtml = "";
    if (alert.file_details && alert.file_details.length > 0) {
        const items = alert.file_details.slice(0, 5).map(f => {
            const t = f.type;
            const color = t === "create" ? "text-neon-green" : t === "rename" ? "text-neon-yellow" : t === "write" ? "text-neon-blue" : "text-neon-red";
            const icon = t === "create" ? "+" : t === "rename" ? "~" : t === "write" ? "w" : "-";
            return `<div class="flex items-center gap-1.5 ${color}"><span class="font-bold">${icon}</span><span class="truncate">${f.path || f.full_path || ""}</span></div>`;
        }).join("");
        fileHtml = `<div class="mt-1.5 pl-4 space-y-0.5 text-[10px] mono">${items}</div>`;
    }

    entry.innerHTML = `
        <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-1.5">
                <svg class="w-3 h-3" style="color:var(--red)" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                <span style="color:var(--red);font-weight:700">ALERT</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-1.5 py-0.5 rounded text-[9px] mono ${isReal ? 'bg-neon-cyan/10 text-neon-cyan' : 'bg-neon-purple/10 text-neon-purple'}">${source}</span>
                <span class="text-gray-500 mono">${time}</span>
            </div>
        </div>
        <div class="text-gray-300 leading-relaxed">${alert.message || ""}</div>
        ${fileHtml}
    `;

    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 20) log.removeChild(log.lastChild);
}

function addEventToFeed(tick) {
    const feed = el("eventFeed");
    if (!feed) return;
    ensurePlaceholderClear(feed, "Waiting");

    const nc = tick.nc || 0, nw = tick.nw || 0, nr = tick.nr || 0, nm = tick.nm || 0, nu = tick.nu || 0;
    const fileEvents = tick.file_events || [];
    const hasActivity = nc > 0 || nw > 0 || nr > 0 || nm > 0 || nu > 0;
    if (!hasActivity && !tick.is_attack && fileEvents.length === 0) return;

    const entry = document.createElement("div");
    const time = new Date().toLocaleTimeString();
    const isAttack = !!tick.is_attack;
    entry.className = `event-new px-2 py-1.5 rounded-lg ${isAttack ? 'bg-red-500/10' : 'hover:bg-cyber-700/30'}`;

    const parts = [];
    if (nc > 0) parts.push(`<span class="text-neon-green">${nc}C</span>`);
    if (nw > 0) parts.push(`<span class="text-neon-blue">${nw}W</span>`);
    if (nr > 0) parts.push(`<span class="text-neon-purple">${nr}R</span>`);
    if (nm > 0) parts.push(`<span class="text-neon-yellow">${nm}N</span>`);
    if (nu > 0) parts.push(`<span class="text-neon-red">${nu}D</span>`);

    let fileHtml = "";
    if (fileEvents.length > 0) {
        const names = fileEvents.slice(0, 4).map(e => {
            const t = e.type;
            const color = t === "create" ? "text-neon-green/60" : t === "rename" ? "text-neon-yellow/60" : t === "write" ? "text-neon-blue/60" : "text-neon-red/60";
            return `<span class="${color} truncate max-w-[120px] inline-block">${e.path || e.full_path || ""}</span>`;
        }).join(", ");
        fileHtml = `<div class="text-[9px] mt-0.5 truncate text-gray-600">${names}</div>`;
    }

    entry.innerHTML = `
        <div class="flex items-center justify-between">
            <span class="text-gray-500 mono">${time}</span>
            <span class="mono text-[10px]">${parts.join(' ')}</span>
            ${isAttack ? '<span style="color:var(--red)" class="text-[9px] font-bold">!</span>' : ''}
        </div>
        ${fileHtml}
    `;

    feed.insertBefore(entry, feed.firstChild);
    while (feed.children.length > 30) feed.removeChild(feed.lastChild);
}

// ──────────────────────────────────────────────────────────────
// Mode management
// ──────────────────────────────────────────────────────────────

function switchMode(mode) {
    // Stop any active run before switching
    if (isActive) {
        if (currentMode === "simulation") socket.emit("stop_simulation");
        else if (currentMode === "monitoring") socket.emit("stop_monitoring");
    }

    currentMode = mode;
    isActive = false;

    if (el("modeSimBtn")) el("modeSimBtn").classList.toggle("mode-btn-active", mode === "simulation");
    if (el("modeRealBtn")) el("modeRealBtn").classList.toggle("mode-btn-active", mode === "real");

    if (el("simControls")) el("simControls").classList.toggle("hidden", mode !== "simulation");
    if (el("realControls")) el("realControls").classList.toggle("hidden", mode !== "real");

    updateButtonStates();
}

function toggleActive() {
    if (currentMode === "simulation") {
        if (isActive) socket.emit("stop_simulation");
        else socket.emit("start_simulation");
    } else if (currentMode === "real") {
        if (isActive) socket.emit("stop_monitoring");
        else socket.emit("start_monitoring", {});
    }
}

function updateButtonStates() {
    const isSim = currentMode === "simulation";
    const btn = isSim ? el("toggleBtn") : el("toggleRealBtn");
    const forceBtn = el("forceBtn");

    if (btn) {
        if (isActive) {
            btn.textContent = isSim ? "STOP" : "STOP MONITORING";
            btn.className = "btn btn-red";
        } else {
            btn.textContent = isSim ? "START" : "START MONITORING";
            btn.className = isSim ? "btn btn-green" : "btn btn-cyan";
        }
    }
    if (forceBtn) forceBtn.disabled = !isActive;

    // Header indicators
    const modeDot = el("modeDot");
    const modeLabel = el("modeLabel");
    if (modeDot && modeLabel) {
        if (isActive) {
            modeDot.className = "status-dot dot-pulse " + (isSim ? "bg-neon-green" : "bg-neon-cyan");
            modeLabel.textContent = isSim ? "Simulation" : "Monitoring";
            modeLabel.className = "text-cyan-300";
        } else {
            modeDot.className = "status-dot bg-gray-500";
            modeLabel.textContent = "Idle";
            modeLabel.className = "text-gray-400";
        }
    }
}

function forceAttack() {
    socket.emit("force_attack", {});
}

function setSpeed(speed) {
    socket.emit("set_speed", { speed: parseFloat(speed) });
    setActiveSpeed(Math.round(speed));
}

function setActiveSpeed(speed) {
    document.querySelectorAll(".speed-btn").forEach(btn => {
        const s = parseInt(btn.dataset.speed, 10);
        btn.classList.toggle("speed-btn-active", s === speed);
    });
}

function resetAll() {
    socket.emit("reset_metrics");
    if (isActive) {
        if (currentMode === "simulation") socket.emit("stop_simulation");
        else socket.emit("stop_monitoring");
    }

    clearChartData();
    liveProb = 0; liveConf = 0; liveFamily = null; liveIsAttack = false;
    if (el("alertLog")) el("alertLog").innerHTML = '<div class="placeholder">No alerts yet.</div>';
    if (el("eventFeed")) el("eventFeed").innerHTML = '<div class="placeholder">Waiting for events...</div>';

    updateMetrics({ total_ticks: 0, total_attacks: 0, total_detections: 0, total_fp: 0, total_fn: 0 });
    updateLiveCM({ tp: 0, fp: 0, tn: 0, fn: 0 });
    updateLiveOps({ nc: 0, nw: 0, nr: 0, nm: 0, nu: 0, user: "--" });
}

// ──────────────────────────────────────────────────────────────
// Socket.IO events
// ──────────────────────────────────────────────────────────────

function setMLStatus(ready, label) {
    const dot = el("mlDot");
    const lbl = el("mlLabel");
    if (dot && lbl) {
        dot.className = "status-dot " + (ready ? "bg-neon-green dot-pulse" : "bg-gray-500");
        lbl.textContent = label;
        lbl.className = ready ? "text-neon-green" : "text-gray-400";
    }
}

socket.on("connect", () => {
    setMLStatus(false, "ML Ready");
    if (el("connBanner")) el("connBanner").classList.add("hidden");
});

socket.on("disconnect", () => {
    setMLStatus(false, "Disconnected");
    if (el("connBanner")) el("connBanner").classList.remove("hidden");
});

socket.on("connect_error", () => {
    setMLStatus(false, "Connection Error");
    if (el("connBanner")) el("connBanner").classList.remove("hidden");
});

socket.on("initial_data", (data) => {
    if (data.models_trained) setMLStatus(true, "ML Ready");

    if (data.model_comparison) updateModelTable({}, data.model_comparison);
    if (data.history) data.history.forEach(tick => { updateChart(tick); updateLiveOps(tick); });
    if (data.metrics) updateMetrics(data.metrics);
    if (data.live_cm) updateLiveCM(data.live_cm);

    if (data.active_mode && data.active_mode !== "idle") {
        currentMode = data.active_mode === "simulation" ? "simulation" : "real";
        switchMode(currentMode);
        isActive = true;
        updateButtonStates();
    } else {
        switchMode("simulation");
    }

    if (data.folders) {
        const names = (data.folders || []).filter(f => f.exists && f.selected).map(f => f.name);
        setText("folderList", names.join(", ") || "None");
    }
});

socket.on("tick_update", (data) => {
    updateChart(data.tick);
    updateLiveOps(data.tick);
    updateGauge(data.tick);
    updateMetrics(data.metrics);
    addEventToFeed(data.tick);
    if (data.live_cm) updateLiveCM(data.live_cm);
    if (data.tick.predictions) updateModelTable(data.tick.predictions, null);

    if (data.alert) {
        if (data.tick.file_events && data.tick.file_events.length > 0) {
            data.alert.file_details = data.tick.file_events;
        }
        addAlert(data.alert);
    }
});

socket.on("mode_status", (data) => {
    if (data.status === "started") {
        isActive = true;
        if (data.mode === "monitoring" && data.folders) {
            const names = data.folders.map(f => {
                const parts = (f.folder || "").split("\\");
                return parts[parts.length - 1];
            }).filter(Boolean);
            setText("folderList", names.join(", ") || "None");
        }
    } else if (data.status === "stopped" || data.status === "error") {
        isActive = false;
    }
    updateButtonStates();
});

socket.on("attack_forced", (data) => {
    if (data && data.success) {
        setText("attackStatus", "ATTACK FORCED");
    }
});

socket.on("speed_updated", (data) => {
    setActiveSpeed(Math.round(data.speed));
});

socket.on("metrics_reset", () => {
    clearChartData();
    liveProb = 0; liveConf = 0; liveFamily = null; liveIsAttack = false;
    if (el("alertLog")) el("alertLog").innerHTML = '<div class="placeholder">No alerts yet.</div>';
    if (el("eventFeed")) el("eventFeed").innerHTML = '<div class="placeholder">Waiting for events...</div>';
    updateMetrics({ total_ticks: 0, total_attacks: 0, total_detections: 0, total_fp: 0, total_fn: 0 });
    updateLiveCM({ tp: 0, fp: 0, tn: 0, fn: 0 });
});

// ──────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initChart();

    const tick = () => {
        const clock = el("clock");
        if (clock) clock.textContent = new Date().toLocaleTimeString();
    };
    tick();
    setInterval(tick, 1000);
});
