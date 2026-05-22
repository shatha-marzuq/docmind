@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500&display=swap');

:root {
  --bg: #ffffff;
  --surface: #faf9f6;
  --ink: #1a1a18;
  --ink2: #5a5a54;
  --ink3: #9a9a92;
  --line: #e8e6e0;
  --line2: #f0ede6;
  --teal: #0f6e56;
  --teal-bg: #e1f5ee;
  --accent: #1D9E75;
  --user-bg: #1a1a18;
  --user-text: #f5f4f0;
  --r-sm: 8px;
  --radius: 16px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  background: var(--bg);
  font-family: 'DM Sans', sans-serif;
  color: var(--ink);
  height: 100%;
}

#root {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── Layout ── */
.layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 16px;
  padding: 12px;
  flex: 1;
  overflow: hidden;
}

/* ── Left Panel ── */
.panel {
  background: var(--surface);
  border: 0.5px solid var(--line);
  border-radius: var(--radius);
  padding: 16px 14px;
  display: flex;
  flex-direction: column;
  gap: 0;
  overflow-y: auto;
}

.logo {
  font-family: 'Instrument Serif', serif;
  font-size: 1.5rem;
  color: var(--ink);
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 3px;
}
.logo-dot {
  width: 8px; height: 8px;
  background: var(--accent);
  border-radius: 50%;
  flex-shrink: 0;
}
.logo-sub {
  font-size: 0.78rem;
  color: var(--ink3);
  font-weight: 300;
  margin-bottom: 14px;
}

.divider { border: none; border-top: 0.5px solid var(--line); margin: 10px 0; }

.sec-label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 1.2px;
  color: var(--ink3);
  text-transform: uppercase;
  margin-bottom: 6px;
  display: block;
}

select {
  width: 100%;
  padding: 6px 8px;
  background: var(--line2);
  border: 0.5px solid var(--line);
  border-radius: var(--r-sm);
  color: var(--ink);
  font-family: 'DM Sans', sans-serif;
  font-size: 0.78rem;
  cursor: pointer;
  outline: none;
}

.adv-toggle {
  background: none;
  border: 0.5px solid var(--line);
  border-radius: var(--r-sm);
  padding: 6px 10px;
  cursor: pointer;
  color: var(--ink2);
  width: 100%;
  text-align: left;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.78rem;
  font-family: 'DM Sans', sans-serif;
}

.adv-body {
  padding: 8px 0 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* ── Dropzone ── */
.dropzone {
  border: 1.5px dashed var(--line);
  border-radius: var(--radius);
  padding: 18px 12px;
  text-align: center;
  cursor: pointer;
  transition: .15s;
  background: var(--bg);
  color: var(--ink3);
  font-size: 0.75rem;
  user-select: none;
}
.dropzone:hover, .dropzone.drag {
  border-color: var(--accent);
  background: var(--teal-bg);
  color: var(--teal);
}

/* ── File strips ── */
.file-strip {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--teal-bg);
  border: 0.5px solid rgba(29,158,117,.2);
  border-radius: var(--r-sm);
  padding: 6px 8px;
  font-size: 0.73rem;
  color: var(--teal);
  overflow: hidden;
}
.file-dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; flex-shrink: 0; }
.file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.file-badge {
  font-size: 9px; padding: 1px 6px;
  background: rgba(29,158,117,.15); color: var(--teal);
  border-radius: 20px; font-weight: 500; letter-spacing: .5px;
  margin-left: auto; flex-shrink: 0;
}

.btn-process {
  background: var(--ink); color: #f5f4f0;
  border: none; border-radius: var(--r-sm);
  padding: 8px; width: 100%;
  cursor: pointer; font-size: 0.8rem;
  margin-top: 8px; transition: .15s;
  font-family: 'DM Sans', sans-serif;
}
.btn-process:hover { background: #333; }
.btn-process:disabled { opacity: .5; cursor: not-allowed; }

/* ── Buttons ── */
.btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: auto; padding-top: 12px; }
.btn {
  border-radius: var(--r-sm); padding: 6px 10px;
  cursor: pointer; font-size: 0.78rem; transition: .15s;
  font-family: 'DM Sans', sans-serif; font-weight: 400;
}
.btn-light { background: var(--bg); color: var(--ink2); border: 0.5px solid var(--line); }
.btn-light:hover { border-color: var(--ink3); color: var(--ink); }
.btn-dark { background: #2d2d2b; color: #f0ede6; border: none; }
.btn-dark:hover { background: #1a1a18; }

/* ── Chat Column ── */
.chat-col {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-header {
  padding: 0 0 12px;
  border-bottom: 0.5px solid var(--line);
  margin-bottom: 14px;
  flex-shrink: 0;
}
.chat-title {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--ink);
  display: flex;
  align-items: center;
  gap: 8px;
}
.chat-doc-tag {
  font-size: 0.72rem;
  color: var(--ink3);
  background: var(--line2);
  padding: 2px 8px;
  border-radius: 20px;
  font-weight: 400;
}

/* ── Messages ── */
.messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-right: 4px;
}

.msg-wrap-user { display: flex; flex-direction: column; align-items: flex-end; }
.msg-wrap-ai { display: flex; flex-direction: column; align-items: flex-start; }
.msg-role {
  font-size: 10px; font-weight: 500; letter-spacing: .8px;
  text-transform: uppercase; color: var(--ink3); margin-bottom: 4px;
}
.bubble-user {
  background: var(--user-bg); color: var(--user-text);
  border-radius: 16px 16px 4px 16px;
  padding: 10px 14px; max-width: 80%;
  font-size: 0.88rem; line-height: 1.65; font-weight: 300;
}
.bubble-ai {
  background: var(--surface); border: 0.5px solid var(--line);
  border-radius: 16px 16px 16px 4px;
  padding: 12px 14px; max-width: 88%;
  font-size: 0.88rem; line-height: 1.7; color: var(--ink);
}

/* ── Citations ── */
.cit-wrap { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 5px; }
.cit-item {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 8px; background: var(--teal-bg);
  border: 0.5px solid rgba(29,158,117,.2); border-radius: 6px;
  font-size: 0.7rem; color: var(--teal);
}
.cit-src { font-weight: 500; font-size: 0.7rem; font-family: monospace; }

/* ── Thinking dots ── */
.thinking { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
.dot {
  width: 5px; height: 5px; background: var(--ink3);
  border-radius: 50%; animation: blink 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,80%,100% { opacity: .2; } 40% { opacity: 1; } }

/* ── Empty state ── */
.empty-state {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 8px;
}

/* ── Error ── */
.err-box {
  background: #fff5f5; border: 0.5px solid #fca5a5;
  border-radius: var(--r-sm); padding: 8px 12px;
  font-size: 0.82rem; color: #991b1b; margin: 6px 0;
  flex-shrink: 0;
}

/* ── Chat input ── */
.chat-input-wrap {
  display: flex; gap: 8px; align-items: flex-end;
  border-top: 0.5px solid var(--line); padding-top: 12px;
  flex-shrink: 0;
}
.chat-input {
  flex: 1; background: var(--surface);
  border: 0.5px solid var(--line); border-radius: 12px;
  padding: 10px 14px; font-size: 0.88rem; color: var(--ink);
  font-family: 'DM Sans', sans-serif; resize: none; outline: none;
  min-height: 42px; max-height: 120px; transition: .15s;
}
.chat-input:focus { border-color: var(--accent); }
.chat-input:disabled { opacity: .5; cursor: not-allowed; }

.send-btn {
  background: var(--ink); color: #f5f4f0; border: none;
  border-radius: 10px; width: 38px; height: 38px;
  cursor: pointer; font-size: 18px; display: flex;
  align-items: center; justify-content: center; flex-shrink: 0; transition: .15s;
}
.send-btn:hover { background: var(--accent); }
.send-btn:disabled { opacity: .4; cursor: not-allowed; }
