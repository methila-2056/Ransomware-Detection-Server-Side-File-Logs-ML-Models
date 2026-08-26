/* ═══════════════════════════════════════════════════════════════
   Ransomware Detection System - Dashboard JS (Upgraded v2)

   Handles:
   - Socket.IO connection + real-time updates
   - Mode switching (Simulation / Real Monitor)
   - Chart.js timeline
   - UI state management
   - Alert logging with file details
   - Live confusion matrix
   - Speed buttons (1x/3x/5x/10x)
   - Real-time event feed with filenames
   ═══════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────
// Socket.IO + State
// ──────────────────────────────────────────────────────────────

const socket = io();
let currentMode = "idle";
let isActive = false;
let chart = null;
let latestComparison = null;

const chartLabels = [];
const chartNC = [];
const chartNR = [];
const chartNU = [];
const chartAttack = [];
const MAX_POINTS = 60;

// ──────────────────────────────────────────────────────────────
// Chart.js
// ──────────────────────────────────────────────────────────────

function initChart() {
    const ctx = document.getElementById("timelineChart").getContext("2d");
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: "Create",
                    data: chartNC,
                    borderColor: "#00ff88",
                    backgroundColor: "rgba(0, 255, 136, 0.08)",
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                },
                {
                    label: "Rename",
                    data: chartNR,
                    borderColor: "#ffcc00",
                    backgroundColor: "rgba(255, 204, 0, 0.04)",
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                },
                {
                    label: "Delete",
                    data: chartNU,
                    borderColor: "#ff3366",
                    backgroundColor: "rgba(255, 51, 102, 0.04)",
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                },
                {
                    label: "Attack Zone",
                    data: chartAttack,
                    borderColor: "rgba(255, 51, 102, 0.5)",
                    backgroundColor: "rgba(255, 51, 102, 0.1)",
                    borderWidth: 1,
                    borderDash: [4, 4],
                    fill: true,
                    tension: 0,
                    pointRadius: 0,
                },
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
                x: {
                    display: true,
                    grid: { color: "rgba(36, 48, 68, 0.3)" },
                    ticks: { color: "#4b5563", font: { family: "JetBrains Mono, monospace", size: 9 }, maxTicksLimit: 12 },
                },
                y: {
                    display: true,
                    beginAtZero: true,
                    grid: { color: "rgba(36, 48, 68, 0.3)" },
                    ticks: { color: "#4b5563", font: { family: "JetBrains Mono, monospace", size: 9 } },
                },
            },
        },
    });
}

function updateChart(tick) {
    chartLabels.push(tick.timestamp + "s");
    chartNC.push(tick.nc);
    chartNR.push(tick.nr);
    chartNU.push(tick.nu);
    chartAttack.push(tick.is_attack ? Math.max(tick.nc, tick.nr, tick.nu) * 1.1 : 0);

    if (chartLabels.length > MAX_POINTS) {
        chartLabels.shift();
        chartNC.shift();
        chartNR.shift();
        chartNU.shift();
        chartAttack.shift();
    }
    chart.update("none");
}

// ──────────────────────────────────────────────────────────────
// UI Updates
// ──────────────────────────────────────────────────────────────

function updateLiveOps(tick) {
    setNum("liveNC", tick.nc);
    setNum("liveNR", tick.nr);
    setNum("liveNU", tick.nu);

    document.getElementById("ncBar").style.width = Math.min(100, (tick.nc / 130) * 100) + "%";
    document.getElementById("nrBar").style.width = Math.min(100, (tick.nr / 130) * 100) + "%";
    document.getElementById("nuBar").style.width = Math.min(100, (tick.nu / 130) * 100) + "%";

    document.getElementById("currentUser").textContent = tick.user || "--";
}

function setNum(id, val) {
    const el = document.getElementById(id);
    const old = el.textContent;
    el.textContent = val;
    if (old !== String(val)) {
        el.classList.remove("num-pulse");
        void el.offsetWidth;
        el.classList.add("num-pulse");
    }
}

function updateGauge(tick) {
    const xgb = tick.predictions?.xgb || {};
    const prob = xgb.probability || 0;
    const conf = xgb.confidence || 0;

    const circumference = 314;
    const arc = document.getElementById("gaugeArc");
    arc.setAttribute("stroke-dasharray", `${prob * circumference} ${circumference}`);

    let color, badgeBg, badgeText, badgeLabel;

    if (prob > 0.7) {
        color = "#ff3366";
        badgeBg = "bg-red-500/15";
        badgeText = "text-red-400";
        badgeLabel = "CRITICAL";
    } else if (prob > 0.4) {
        color = "#ffcc00";
        badgeBg = "bg-yellow-500/15";
        badgeText = "text-yellow-400";
        badgeLabel = "WARNING";
    } else {
        color = "#00ff88";
        badgeBg = "bg-green-500/15";
        badgeText = "text-green-400";
        badgeLabel = "NO THREAT";
    }

    arc.setAttribute("stroke", color);
    document.getElementById("gaugeValue").textContent = (prob * 100).toFixed(1) + "%";
    document.getElementById("gaugeValue").style.color = color;
    document.getElementById("gaugeLabel").textContent = prob > 0.7 ? "CRITICAL" : prob > 0.4 ? "WARNING" : "SAFE";
    document.getElementById("gaugeLabel").style.color = color;

    const badge = document.getElementById("threatBadge");
    badge.className = `inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold ${badgeBg} ${badgeText} border border-current/20`;
    badge.innerHTML = `<div class="w-1.5 h-1.5 rounded-full bg-current ${prob > 0.5 ? 'badge-pulse' : ''}"></div>${badgeLabel}`;

    document.getElementById("confidenceVal").textContent = conf > 0 ? conf.toFixed(1) + "%" : "--";
    document.getElementById("attackFamily").textContent = tick.family || "None";
    document.getElementById("attackDuration").textContent = tick.is_attack ? "Active" : "0s";

    const main = document.querySelector("main");
    if (prob > 0.7) {
        main.classList.add("glow-red");
    } else {
        main.classList.remove("glow-red");
    }
}

function updateMetrics(m) {
    document.getElementById("metricTicks").textContent = m.total_ticks;
    document.getElementById("metricAttacks").textContent = m.total_attacks;
    document.getElementById("metricDetections").textContent = m.total_detections;
    document.getElementById("metricFP").textContent = m.total_fp;
    document.getElementById("metricFN").textContent = m.total_fn;

    const detRate = m.total_attacks > 0 ? ((m.total_detections / m.total_attacks) * 100) : 0;
    document.getElementById("metricDetRate").textContent = detRate.toFixed(1) + "%";
    document.getElementById("detRateBar").style.width = detRate + "%";
}

function updateLiveCM(cm) {
    if (!cm) return;
    document.getElementById("liveCM_TP").textContent = cm.tp || 0;
    document.getElementById("liveCM_FP").textContent = cm.fp || 0;
    document.getElementById("liveCM_TN").textContent = Math.max(0, cm.tn || 0);
    document.getElementById("liveCM_FN").textContent = cm.fn || 0;
}

function updateTestCM(comparison) {
    if (!comparison || comparison.length === 0) return;
    const xgb = comparison.find(m => m.name === "XGBoost");
    if (xgb) {
        document.getElementById("testCM_TP").textContent = xgb.tp;
        document.getElementById("testCM_FP").textContent = xgb.fp;
        document.getElementById("testCM_TN").textContent = xgb.tn;
        document.getElementById("testCM_FN").textContent = xgb.fn;
    }
}

function updateModelTable(predictions, comparison) {
    if (comparison && comparison.length > 0) latestComparison = comparison;
    const cmp = (comparison && comparison.length > 0) ? comparison : latestComparison;
    const tbody = document.getElementById("modelTableBody");

    if (cmp && cmp.length > 0) {
        let bestIdx = 0;
        let bestSens = 0;
        cmp.forEach((m, i) => { if (m.sensitivity > bestSens) { bestSens = m.sensitivity; bestIdx = i; } });

        let html = "";
        cmp.forEach((m, i) => {
            const isBest = i === bestIdx;
            const pred = (m.key && predictions && predictions[m.key]) ? predictions[m.key] : (predictions ? predictions[Object.keys(predictions)[i]] : null);
            const predText = pred
                ? (pred.prediction === 1 ? '<span class="text-red-400 font-bold">ATTACK</span>' : '<span class="text-green-400">SAFE</span>')
                : '<span class="text-gray-600">--</span>';

            html += `<tr class="${isBest ? 'best-row' : ''}">
                <td class="py-2 pr-3 font-medium text-gray-200">${m.name}</td>
                <td class="text-center tabular-nums">${(m.accuracy * 100).toFixed(1)}%</td>
                <td class="text-center tabular-nums ${m.sensitivity > 0.9 ? 'text-neon-green' : ''}">${(m.sensitivity * 100).toFixed(1)}%</td>
                <td class="text-center tabular-nums">${(m.f1_score * 100).toFixed(1)}%</td>
                <td class="text-center tabular-nums text-gray-500">${m.prediction_latency_ms.toFixed(2)}ms</td>
                <td class="text-center text-[10px]">${predText}</td>
            </tr>`;
        });
        tbody.innerHTML = html;

        updateTestCM(cmp);
    }
}

// ──────────────────────────────────────────────────────────────
// Alerts + Events with File Details
// ──────────────────────────────────────────────────────────────

function addAlert(alert) {
    const log = document.getElementById("alertLog");
    if (log.children.length === 1 && log.children[0].textContent.includes("No alerts")) {
        log.innerHTML = "";
    }

    const entry = document.createElement("div");
    entry.className = "alert-enter p-2.5 rounded-xl bg-red-500/8 border border-red-500/15 text-[11px]";

    const time = new Date().toLocaleTimeString();
    const source = alert.source === "real" ? "REAL" : "SIM";

    // Build file details HTML if available
    let fileHtml = "";
    if (alert.file_details && alert.file_details.length > 0) {
        const fileItems = alert.file_details.slice(0, 5).map(f => {
            const color = f.type === "create" ? "text-neon-green" : f.type === "rename" ? "text-neon-yellow" : "text-neon-red";
            const icon = f.type === "create" ? "+" : f.type === "rename" ? "~" : "-";
            return `<div class="flex items-center gap-1.5 ${color}"><span class="font-bold">${icon}</span><span class="truncate">${f.path}</span></div>`;
        }).join("");
        fileHtml = `<div class="mt-1.5 pl-4 space-y-0.5 text-[10px] font-mono">${fileItems}</div>`;
    }

    entry.innerHTML = `
        <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-1.5">
                <svg class="w-3 h-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                <span class="text-red-400 font-bold">ALERT</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-1.5 py-0.5 rounded text-[9px] font-mono ${source === 'REAL' ? 'bg-neon-cyan/10 text-neon-cyan' : 'bg-neon-purple/10 text-neon-purple'}">${source}</span>
                <span class="text-gray-500 font-mono">${time}</span>
            </div>
        </div>
        <div class="text-gray-300 leading-relaxed">${alert.message}</div>
        ${fileHtml}
    `;

    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 20) log.removeChild(log.lastChild);
}

function addEventToFeed(tick) {
    const feed = document.getElementById("eventFeed");

    if (feed.children.length === 1 && feed.children[0].textContent.includes("Waiting")) {
        feed.innerHTML = "";
    }

    const nc = tick.nc, nr = tick.nr, nu = tick.nu;
    const hasActivity = nc > 0 || nr > 0 || nu > 0;
    const fileEvents = tick.file_events || [];
    if (!hasActivity && !tick.is_attack && fileEvents.length === 0) return;

    const entry = document.createElement("div");
    const time = new Date().toLocaleTimeString();
    const isAttack = tick.is_attack;

    entry.className = `event-new px-2 py-1.5 rounded-lg ${isAttack ? 'bg-red-500/8' : 'hover:bg-cyber-700/30'}`;

    // Count summary
    const parts = [];
    if (nc > 0) parts.push(`<span class="text-neon-green">${nc}C</span>`);
    if (nr > 0) parts.push(`<span class="text-neon-yellow">${nr}R</span>`);
    if (nu > 0) parts.push(`<span class="text-neon-red">${nu}D</span>`);

    // File names (if real monitoring)
    let fileHtml = "";
    if (fileEvents.length > 0) {
        const names = fileEvents.slice(0, 4).map(e => {
            const color = e.type === "create" ? "text-neon-green/60" : e.type === "rename" ? "text-neon-yellow/60" : "text-neon-red/60";
            return `<span class="${color} truncate max-w-[120px] inline-block">${e.path}</span>`;
        }).join(", ");
        fileHtml = `<div class="text-[9px] mt-0.5 truncate text-gray-600">${names}</div>`;
    }

    entry.innerHTML = `
        <div class="flex items-center justify-between">
            <span class="text-gray-500 font-mono">${time}</span>
            <span class="font-mono text-[10px]">${parts.join(' ')}</span>
            ${isAttack ? '<span class="text-red-400 text-[9px] font-bold">!</span>' : ''}
        </div>
        ${fileHtml}
    `;

    feed.insertBefore(entry, feed.firstChild);
    while (feed.children.length > 30) feed.removeChild(feed.lastChild);
}

// ──────────────────────────────────────────────────────────────
// Mode Management
// ──────────────────────────────────────────────────────────────

function switchMode(mode) {
    if (isActive) {
        if (currentMode === "simulation") socket.emit("stop_simulation");
        else if (currentMode === "monitoring") socket.emit("stop_monitoring");
    }

    currentMode = mode;
    isActive = false;

    document.getElementById("modeSimBtn").className = "mode-btn px-5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 " + (mode === "simulation" ? "mode-btn-active" : "text-gray-500 hover:text-gray-300");
    document.getElementById("modeRealBtn").className = "mode-btn px-5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 " + (mode === "real" ? "mode-btn-active" : "text-gray-500 hover:text-gray-300");

    document.getElementById("simControls").classList.toggle("hidden", mode !== "simulation");
    document.getElementById("realControls").classList.toggle("hidden", mode !== "real");

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
    const btn = currentMode === "simulation" ? document.getElementById("toggleBtn") : document.getElementById("toggleRealBtn");
    const forceBtn = document.getElementById("forceBtn");

    if (isActive) {
        if (currentMode === "simulation") {
            btn.textContent = "STOP";
            btn.className = "px-5 py-2 rounded-xl text-xs font-bold transition-all duration-200 bg-neon-red/80 text-white hover:bg-neon-red hover:shadow-lg hover:shadow-neon-red/20 active:scale-95";
        } else {
            btn.textContent = "STOP MONITORING";
            btn.className = "px-5 py-2 rounded-xl text-xs font-bold transition-all duration-200 bg-neon-red/80 text-white hover:bg-neon-red hover:shadow-lg hover:shadow-neon-red/20 active:scale-95";
        }
        if (forceBtn) forceBtn.disabled = false;
    } else {
        if (currentMode === "simulation") {
            btn.textContent = "START";
            btn.className = "px-5 py-2 rounded-xl text-xs font-bold transition-all duration-200 bg-neon-green/90 text-cyber-950 hover:bg-neon-green hover:shadow-lg hover:shadow-neon-green/20 active:scale-95";
        } else {
            btn.textContent = "START MONITORING";
            btn.className = "px-5 py-2 rounded-xl text-xs font-bold transition-all duration-200 bg-neon-cyan/90 text-cyber-950 hover:bg-neon-cyan hover:shadow-lg hover:shadow-neon-cyan/20 active:scale-95";
        }
        if (forceBtn) forceBtn.disabled = true;
    }

    // Header indicators
    const modeDot = document.getElementById("modeDot");
    const modeLabel = document.getElementById("modeLabel");
    if (isActive) {
        modeDot.className = "w-2 h-2 rounded-full " + (currentMode === "simulation" ? "bg-neon-green dot-pulse" : "bg-neon-cyan dot-pulse");
        modeLabel.textContent = currentMode === "simulation" ? "Simulation" : "Monitoring";
        modeLabel.className = "text-[11px] font-medium " + (currentMode === "simulation" ? "text-neon-green" : "text-neon-cyan");
    } else {
        modeDot.className = "w-2 h-2 rounded-full bg-gray-500";
        modeLabel.textContent = "Idle";
        modeLabel.className = "text-[11px] text-gray-400 font-medium";
    }
}

function forceAttack() {
    socket.emit("force_attack", {});
}

function setSpeed(speed) {
    socket.emit("set_speed", { speed: parseFloat(speed) });
    // Update button active states
    document.querySelectorAll(".speed-btn").forEach(btn => {
        const s = parseInt(btn.dataset.speed);
        if (s === speed) {
            btn.className = "speed-btn px-2 py-1 rounded text-[10px] font-bold bg-neon-blue/20 text-neon-blue border border-neon-blue/30 transition-all";
        } else {
            btn.className = "speed-btn px-2 py-1 rounded text-[10px] font-bold text-gray-500 border border-cyber-600/30 hover:text-gray-300 transition-all";
        }
    });
}

function resetAll() {
    socket.emit("reset_metrics");
    if (isActive) {
        if (currentMode === "simulation") socket.emit("stop_simulation");
        else socket.emit("stop_monitoring");
    }
    chartLabels.length = 0;
    chartNC.length = 0;
    chartNR.length = 0;
    chartNU.length = 0;
    chartAttack.length = 0;
    chart.update("none");

    document.getElementById("alertLog").innerHTML = '<div class="text-[11px] text-gray-600 text-center py-6">No alerts yet.</div>';
    document.getElementById("eventFeed").innerHTML = '<div class="text-gray-600 text-center py-6">Waiting for events...</div>';

    updateMetrics({ total_ticks: 0, total_attacks: 0, total_detections: 0, total_fp: 0, total_fn: 0 });
    updateLiveCM({ tp: 0, fp: 0, tn: 0, fn: 0 });
}

// ──────────────────────────────────────────────────────────────
// Socket.IO Events
// ──────────────────────────────────────────────────────────────

socket.on("connect", () => {
    document.getElementById("mlDot").className = "w-2 h-2 rounded-full bg-neon-green dot-pulse";
    document.getElementById("mlLabel").textContent = "ML Ready";
    document.getElementById("mlLabel").className = "text-[11px] text-neon-green font-medium";
});

socket.on("disconnect", () => {
    document.getElementById("mlDot").className = "w-2 h-2 rounded-full bg-gray-500";
    document.getElementById("mlLabel").textContent = "Disconnected";
    document.getElementById("mlLabel").className = "text-[11px] text-gray-400 font-medium";
});

socket.on("initial_data", (data) => {
    if (data.models_trained) {
        document.getElementById("mlDot").className = "w-2 h-2 rounded-full bg-neon-green";
        document.getElementById("mlLabel").textContent = "ML Ready";
        document.getElementById("mlLabel").className = "text-[11px] text-neon-green font-medium";
    }
    if (data.model_comparison) updateModelTable({}, data.model_comparison);
    if (data.history) data.history.forEach(tick => { updateChart(tick); updateLiveOps(tick); });
    if (data.metrics) updateMetrics(data.metrics);
    if (data.live_cm) updateLiveCM(data.live_cm);
    if (data.active_mode && data.active_mode !== "idle") {
        currentMode = data.active_mode === "simulation" ? "simulation" : "real";
        isActive = true;
        switchMode(currentMode);
        isActive = true;
        updateButtonStates();
    }
    if (data.folders) {
        const names = data.folders.filter(f => f.exists && f.selected).map(f => f.name);
        document.getElementById("folderList").textContent = names.join(", ") || "None";
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
        // Attach file details to alert for display
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
                const parts = f.folder.split("\\");
                return parts[parts.length - 1];
            });
            document.getElementById("folderList").textContent = names.join(", ");
        }
    } else if (data.status === "stopped" || data.status === "error") {
        isActive = false;
    }
    updateButtonStates();
});

socket.on("attack_forced", (data) => {
    console.log("Attack forced:", data);
});

socket.on("speed_updated", (data) => {
    const speedVal = document.getElementById("speedVal");
    if (speedVal) speedVal.textContent = data.speed + "x";
});

socket.on("metrics_reset", () => {
    chartLabels.length = 0;
    chartNC.length = 0;
    chartNR.length = 0;
    chartNU.length = 0;
    chartAttack.length = 0;
    chart.update("none");
    document.getElementById("alertLog").innerHTML = '<div class="text-[11px] text-gray-600 text-center py-6">No alerts yet.</div>';
    document.getElementById("eventFeed").innerHTML = '<div class="text-gray-600 text-center py-6">Waiting for events...</div>';
    updateMetrics({ total_ticks: 0, total_attacks: 0, total_detections: 0, total_fp: 0, total_fn: 0 });
    updateLiveCM({ tp: 0, fp: 0, tn: 0, fn: 0 });
});

// ──────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    setInterval(() => {
        document.getElementById("clock").textContent = new Date().toLocaleTimeString();
    }, 1000);
    document.getElementById("clock").textContent = new Date().toLocaleTimeString();
});
