const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getStatus() {
  const res = await fetch(`${BASE}/status`);
  if (!res.ok) throw new Error("Failed to get status");
  return res.json();
}

export async function uploadFiles(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Upload failed");
  return data;
}

export async function askQuestion({ question, searchMode, topK }) {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      search_mode: searchMode,
      top_k: topK,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Request failed");
  return data;
}

export async function resetAll() {
  const res = await fetch(`${BASE}/reset`, { method: "POST" });
  if (!res.ok) throw new Error("Reset failed");
  return res.json();
}
