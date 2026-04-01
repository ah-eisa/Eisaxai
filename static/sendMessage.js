// --- 1. VANTA BACKGROUND LOGIC (Dynamic) ---
let vantaEffect = null;
let currentAttachment = null;

async function handleFileUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const ta = document.getElementById('prompt');
    const oldPlaceholder = ta.placeholder;
    ta.placeholder = "Uploading " + file.name + "...";
    ta.disabled = true;

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'received') {
            currentAttachment = { id: data.file_id, name: data.filename };
            showAttachmentPreview(data.filename);
        }
    } catch (e) {
        alert("Upload failed: " + e);
    } finally {
        ta.placeholder = oldPlaceholder;
        ta.disabled = false;
        input.value = '';
    }
}

function showAttachmentPreview(name) {
    const preview = document.getElementById('attachment-preview');
    const nameEl = document.getElementById('attachment-name');
    const wrapper = document.getElementById('input-wrapper');
    if (preview && nameEl) {
        nameEl.textContent = name;
        preview.style.display = 'flex';
        if (wrapper) wrapper.style.borderRadius = '0 0 14px 14px';
    }
}

function clearAttachment() {
    currentAttachment = null;
    const preview = document.getElementById('attachment-preview');
    const wrapper = document.getElementById('input-wrapper');
    if (preview) {
        preview.style.display = 'none';
        if (wrapper) wrapper.style.borderRadius = '18px';
    }
}

function initVanta(isLight) {
    const colorBg = isLight ? 0xf8fafc : 0x0f172a; // Slate 900
    const colorLine = isLight ? 0xcdd5e1 : 0x3b82f6; // Blue 500 (Vibrant)

    if (vantaEffect) {
        vantaEffect.setOptions({
            backgroundColor: colorBg,
            color: colorLine
        });
    } else {
        try {
            vantaEffect = VANTA.NET({
                el: "#vanta-bg",
                mouseControls: true,
                touchControls: true,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                points: 12.00,   // More points
                maxDistance: 22.00,
                spacing: 18.00,
                backgroundColor: colorBg,
                color: colorLine
            });
        } catch (e) { console.log("Vanta Error:", e); }
    }
}

// --- 2. THEME LOGIC ---
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    const isLight = document.body.classList.contains('light-theme');
    localStorage.setItem('ae_theme', isLight ? 'light' : 'dark');
    initVanta(isLight);
}

// --- 3. SESSION LOGIC (API Based) ---
function uuidv4() { return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8); return v.toString(16); }); }

// We keep "mode" locally because it's a UI preference, but session data is now remote.
let activeAgentMode = localStorage.getItem('clawdbot_mode') || 'assistant';
let currentSessionId = localStorage.getItem('ae_current_sid') || uuidv4();
let viewArchived = false;

// Initialize Session ID if needed
if (!currentSessionId) {
    currentSessionId = uuidv4();
    localStorage.setItem('ae_current_sid', currentSessionId);
}

function switchAgent(mode) {
    if (mode === activeAgentMode) return;
    activeAgentMode = mode;
    localStorage.setItem('clawdbot_mode', mode);

    // For simplicity, switching agents keeps the current session but changes system behavior.
    // Or we could start a new session. Let's keep current session to allow "Consultant switching".

    updateAgentUI();

    // Notify user of switch
    appendUI('bot', `Switched to **${mode === 'investment' ? 'Investment' : 'General'} Assistant**. How can I help?`);
}

function updateAgentUI() {
    const modeText = document.getElementById('mode-text');
    const modeDot = document.getElementById('mode-dot');

    if (modeText) {
        modeText.textContent = activeAgentMode === 'investment' ?
            'Investment Assistant' : 'General Assistant';
    }
    if (modeDot) {
        modeDot.style.background = activeAgentMode === 'investment' ?
            '#10b981' : 'var(--primary)';
    }
}

window.confirmSwitchToMode = function (targetMode) {
    switchAgent(targetMode);
    window.pendingModeSwitch = null;
};

// --- API BASED RENERING ---

async function renderSessions() {
    const container = document.getElementById('session-list');
    container.innerHTML = '<div style="padding:10px; opacity:0.5; font-size:0.8rem;">Loading...</div>';

    try {
        const list = await window.fetchHistory("admin"); // Default user
        container.innerHTML = '';

        if (!list || list.length === 0) {
            container.innerHTML = '<div style="padding:10px; opacity:0.5; font-size:0.8rem;">No past sessions</div>';
            return;
        }

        // Grouping Logic
        const groups = { 'Today': [], 'Older': [] };
        const now = new Date();

        list.forEach(s => {
            // Check if archived (need to implement archive logic on backend or generic JSON store)
            // For now, assume all backend sessions are "active" unless we add an 'archived' column.
            // We'll skip archive filtering for MVP history.

            const d = new Date(s.created_at + "Z"); // Ensure UTC if needed, or backend sends ISO
            // Simple date check
            if (d.toDateString() === now.toDateString()) {
                groups['Today'].push(s);
            } else {
                groups['Older'].push(s);
            }
        });

        for (const [label, items] of Object.entries(groups)) {
            if (items.length === 0) continue;

            const head = document.createElement('div');
            head.className = 'group-header';
            head.textContent = label;
            container.appendChild(head);

            items.forEach(s => {
                const el = document.createElement('div');
                el.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
                el.onclick = () => loadSession(s.id);

                el.innerHTML = `
                    <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;">${s.title}</span>
                    <div class="session-actions">
                        <span class="action-icon del" title="Delete" onclick="deleteSession(event, '${s.id}')">✕</span>
                    </div>
                `;
                container.appendChild(el);
            });
        }
    } catch (e) {
        console.error("History Error:", e);
        container.innerHTML = '<div style="padding:10px; color:red; font-size:0.8rem;">Error loading history</div>';
    }
}

async function deleteSession(e, sid) {
    e.stopPropagation();
    if (!confirm("Delete this conversation?")) return;

    await window.deleteSession(sid);

    if (sid === currentSessionId) {
        startNewSession();
    } else {
        renderSessions();
    }
}

async function loadSession(sid) {
    currentSessionId = sid;
    localStorage.setItem('ae_current_sid', sid);

    // Update active highlight
    renderSessions();

    const box = document.getElementById('chat-container');
    box.innerHTML = '<div class="msg bot"><div class="avatar">AI</div><div class="bubble">Loading conversation...</div></div>';

    try {
        const msgs = await window.loadSessionMessages(sid);
        box.innerHTML = ''; // Clear loading

        if (msgs.length === 0) {
            showGreeting();
        } else {
            msgs.forEach(m => {
                // Map backend roles to frontend classes
                const role = (m.role === 'assistant' || m.role === 'bot') ? 'bot' : 'user';
                appendUI(role, m.content);
            });
        }

        // Scroll to bottom
        setTimeout(() => box.scrollTo({ top: box.scrollHeight }), 100);

    } catch (e) {
        box.innerHTML = `<div class="msg bot"><div class="bubble">Error loading chat: ${e}</div></div>`;
    }
}

function startNewSession() {
    currentSessionId = uuidv4();
    localStorage.setItem('ae_current_sid', currentSessionId);
    showGreeting();
    renderSessions(); // Refresh list to remove active highlight
}

function showGreeting() {
    const box = document.getElementById('chat-container');
    const greeting = activeAgentMode === 'investment'
        ? `<div class="msg bot"><div class="avatar">AI</div><div class="bubble md-content"><strong>Hello.</strong><br>I'm your Investment Assistant. What would you like to analyze today?</div></div>`
        : `<div class="msg bot"><div class="avatar">AI</div><div class="bubble md-content"><strong>Hello.</strong><br>I'm **EisaX**, your AI financial assistant. How can I help you today?</div></div>`;
    box.innerHTML = greeting;
}

function toggleArchivedView() {
    // Backend doesn't support archiving yet, so we just toggle UI label
    alert("Archiving not yet supported on backend.");
}

// --- 4. CHAT FUNCTIONS ---

async function send() {
    const ta = document.getElementById('prompt');
    let text = ta.value.trim();
    if (!text && !currentAttachment) return;

    if (!text && currentAttachment) text = "Analyze the attached file.";

    let displayText = text;
    if (currentAttachment) {
        displayText += `\n\n📎 *Attached: ${currentAttachment.name}*`;
    }

    ta.value = ''; autoGrow(ta);

    // UI Hider for internal commands
    const isInternal = text.includes("internal_command") || text.includes("system_function");
    if (!isInternal) {
        appendUI('user', displayText);
    }

    const fileIdToSend = currentAttachment ? currentAttachment.id : null;
    clearAttachment();

    // Show Loading
    const loadingId = 'load-' + Date.now();
    const box = document.getElementById('chat-container');
    const loadEl = document.createElement('div');
    loadEl.id = loadingId; loadEl.className = 'msg bot';
    loadEl.innerHTML = '<div class="avatar">AI</div><div class="bubble" style="color:var(--primary)">Processing...</div>';
    box.appendChild(loadEl);
    requestAnimationFrame(() => { box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' }); });

    try {
        const payload = {
            message: text,
            prompt: text,
            session_id: currentSessionId, // Send explicit session ID
            user_id: "admin", // Explicit user ID
            settings: {
                session_id: currentSessionId,
                assistant_type: activeAgentMode,
                mode: activeAgentMode,
                memory: true,
                active_file_id: fileIdToSend
            }
        };

        const headers = {
            'Content-Type': 'application/json',
            'access-token': 'EisaX_2026_Secure'
        };

        // ── Primary path: SSE streaming (/v1/chat/stream) ───────────────────
        let streamedReply = '';
        let streamedMeta = {};
        let streamSucceeded = false;

        try {
            const streamRes = await fetch('/v1/chat/stream', {
                method: 'POST',
                headers,
                body: JSON.stringify(payload)
            });
            if (!streamRes.ok) throw new Error("Streaming API Error: " + streamRes.status);
            if (!streamRes.body || !streamRes.body.getReader) {
                throw new Error("Streaming not supported by this browser.");
            }

            const reader = streamRes.body.getReader();
            const decoder = new TextDecoder('utf-8');
            const loadingEl = document.getElementById(loadingId);
            const loadingBubble = loadingEl ? loadingEl.querySelector('.bubble') : null;
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const packets = buffer.split('\n\n');
                buffer = packets.pop() || '';

                for (const packet of packets) {
                    const lines = packet
                        .split('\n')
                        .filter(line => line.startsWith('data:'))
                        .map(line => line.slice(5).trim());

                    if (!lines.length) continue;
                    const dataStr = lines.join('\n');
                    if (dataStr === '[DONE]') continue;

                    let evt;
                    try {
                        evt = JSON.parse(dataStr);
                    } catch (_parseErr) {
                        continue;
                    }

                    if (evt.type === 'status') {
                        if (loadingBubble && !streamedReply) {
                            loadingBubble.style.color = 'var(--primary)';
                            loadingBubble.textContent = evt.text || 'Processing...';
                        }
                    } else if (evt.type === 'token') {
                        const chunk = evt.text || '';
                        if (!chunk) continue;
                        streamedReply += chunk;
                        if (loadingBubble) {
                            loadingBubble.style.color = '';
                            loadingBubble.classList.add('md-content');
                            loadingBubble.innerHTML = marked.parse(streamedReply);
                        }
                        requestAnimationFrame(() => {
                            box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
                        });
                    } else if (evt.type === 'done') {
                        if (typeof evt.text === 'string') {
                            try { streamedMeta = JSON.parse(evt.text); } catch (_ignore) { streamedMeta = {}; }
                        } else if (evt.text && typeof evt.text === 'object') {
                            streamedMeta = evt.text;
                        }
                    } else if (evt.type === 'error') {
                        throw new Error(evt.text || 'Streaming failed');
                    }
                }
            }

            streamSucceeded = streamedReply.trim().length > 0;
        } catch (streamErr) {
            console.warn('[chat] Streaming failed, falling back to /v1/chat:', streamErr);
        }

        if (streamSucceeded) {
            if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
            appendUI('bot', streamedReply, null, null, streamedMeta);
            renderSessions();
            return;
        }

        // ── Fallback path: classic one-shot (/v1/chat) ───────────────────────
        const res = await fetch('/v1/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error("API Error: " + res.status);

        const data = await res.json();
        if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
        const reply = data.reply || (data.choices && data.choices[0].message.content) || JSON.stringify(data);

        appendUI('bot', reply, null, data.type, data.data);
        renderSessions();

    } catch (e) {
        if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
        appendUI('bot', '**Error:** ' + e);
    }
}

// Include appendUI, printReport, runAnalysis (Same as before, abbreviated here for clarity but included in file write)
function appendUI(role, text, extraHtml = null, messageType = null, messageData = null) {
    if (!text && !messageData) return;
    const box = document.getElementById('chat-container');
    const d = document.createElement('div');
    d.className = `msg ${role}`;

    // Parse Markdown
    const content = role === 'bot' ? marked.parse(text || "") : text;

    // ... (rest of appendUI logic mostly buttons) ...
    // Re-implementing button logic simply:

    let bubbleContent = content;

    if (role === 'bot') {
        // Download Button
        if (messageData && (messageData.download_url || messageData.url)) {
            const url = messageData.download_url || messageData.url;
            bubbleContent += `<br><a href="${url}" class="print-report-btn" target="_blank" style="margin-top:10px">Download File</a>`;
        }
        // Run Button
        if (messageData && messageData.show_run_button) {
            bubbleContent += `<br><button class="print-report-btn" onclick="runAnalysis()">Run Analysis</button>`;
        }
        // Print Button (simple check)
        if (messageData && messageData.printable) {
            const safeId = 'print-' + Date.now();
            window[safeId] = text;
            bubbleContent += `<br><button class="print-report-btn" onclick="printReport(window['${safeId}'], '${messageData.title || 'Report'}')">Print Report</button>`;
        }
    }

    // Add export bar for analysis reports
    let exportBar = '';
    if (role === 'bot' && isAnalysisReport(text)) {
        const safeText = encodeURIComponent(text || '').substring(0, 100); // just for reference
        exportBar = `<div class="export-bar">
            <button class="export-btn primary" onclick="exportPDF(null,'en',this)">📄 Download Report</button>
            <button class="export-btn arabic" onclick="exportPDF(null,'ar',this)">📄 تقرير عربي</button>
            <button class="export-btn" onclick="shareReport(null,this)">📤 Share</button>
            <button class="export-btn" onclick="window.print()">🖨️ Print</button>
        </div>`;
    }

    d.innerHTML = `<div class="avatar">${role === 'bot' ? 'AI' : 'ME'}</div><div class="bubble md-content">${bubbleContent}${exportBar}</div>`;
    box.appendChild(d);
    requestAnimationFrame(() => { box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' }); });
}

function autoGrow(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 150) + 'px'; }
document.getElementById('prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const floatingToggle = document.getElementById('floating-toggle');
    sidebar.classList.toggle('collapsed');
    if (sidebar.classList.contains('collapsed')) {
        floatingToggle.classList.add('visible');
    } else {
        floatingToggle.classList.remove('visible');
    }
}

// ── Export / Share toolbar ────────────────────────────────────────────
function isAnalysisReport(text) {
    if (!text || text.length < 400) return false;
    const kw = ['EisaX Intelligence Report', 'MEMORANDUM', 'Quality Score',
                'Investment Analysis', 'تقرير', 'تحليل', 'توصية', 'RSI', 'MACD'];
    return kw.some(k => text.includes(k));
}

async function exportPDF(text, lang, btnEl) {
    if (btnEl) { btnEl.disabled = true; btnEl.classList.add('loading'); btnEl.textContent = '⏳ Generating...'; }
    try {
        const messages = getConversationMessages();
        const res = await fetch('/v1/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'access-token': 'EisaX_2026_Secure' },
            body: JSON.stringify({ format: lang === 'ar' ? 'pdf_ar' : 'pdf', messages, title: 'EisaX Report' })
        });
        const data = await res.json();
        if (data.download_url) {
            const a = document.createElement('a');
            a.href = data.download_url;
            a.download = data.filename || 'EisaX_Report.pdf';
            a.target = '_blank';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            if (btnEl) { btnEl.disabled = false; btnEl.classList.remove('loading'); btnEl.innerHTML = lang === 'ar' ? '📄 تقرير عربي' : '📄 Download Report'; }
        } else {
            throw new Error(data.detail || 'Export failed');
        }
    } catch(e) {
        alert('Export error: ' + e.message);
        if (btnEl) { btnEl.disabled = false; btnEl.classList.remove('loading'); btnEl.innerHTML = lang === 'ar' ? '📄 تقرير عربي' : '📄 Download Report'; }
    }
}

function getConversationMessages() {
    const msgs = [];
    document.querySelectorAll('.msg').forEach(el => {
        const role = el.classList.contains('bot') ? 'assistant' : 'user';
        const bubble = el.querySelector('.bubble');
        if (bubble) msgs.push({ role, content: bubble.innerText || bubble.textContent || '' });
    });
    return msgs;
}

async function shareReport(text, btnEl) {
    const url = window.location.href;
    const shareText = text ? text.substring(0, 300) + '...' : 'EisaX Investment Report';
    if (navigator.share) {
        try {
            await navigator.share({ title: 'EisaX Report', text: shareText, url });
            return;
        } catch(e) {}
    }
    try {
        await navigator.clipboard.writeText(url);
        if (btnEl) {
            const orig = btnEl.innerHTML;
            btnEl.innerHTML = '✅ Link Copied!';
            setTimeout(() => { btnEl.innerHTML = orig; }, 2000);
        }
    } catch(e) {
        prompt('Copy this link:', url);
    }
}

// Print logic
function printReport(markdownContent, title) {
    const printWindow = window.open('', '_blank', 'width=900,height=700');
    const htmlContent = marked.parse(markdownContent);
    const htmlDoc = `<!DOCTYPE html><html><head><title>${title}</title><style>body{font-family:sans-serif;padding:40px;}</style></head><body><h1>${title}</h1>${htmlContent}<script>setTimeout(()=>{window.print()},500)</script></body></html>`;
    printWindow.document.write(htmlDoc);
    printWindow.document.close();
}

function runAnalysis() {
    const ta = document.getElementById('prompt');
    ta.value = "proceed with comprehensive analysis";
    send();
}

// --- INIT ---
document.addEventListener('DOMContentLoaded', () => {
    // Config Marked
    if (window.marked) {
        const renderer = new marked.Renderer();
        renderer.link = (href, title, text) => `<a href="${href}" target="_blank">${text}</a>`;
        marked.use({ renderer });
    }

    const savedTheme = localStorage.getItem('ae_theme');
    if (savedTheme === 'light') document.body.classList.add('light-theme');
    initVanta(savedTheme === 'light');

    updateAgentUI();

    // Load initial session
    if (currentSessionId) {
        loadSession(currentSessionId);
    } else {
        startNewSession();
    }
});
