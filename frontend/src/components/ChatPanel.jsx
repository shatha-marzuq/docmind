import { useState, useRef } from "react";

const SUPPORTED = [".pdf", ".txt", ".docx", ".md"];

export default function LeftPanel({
  docsLoaded,
  fileNames,
  searchMode,
  setSearchMode,
  topK,
  setTopK,
  onProcess,
  onClearChat,
  onReset,
  processing,
}) {
  const [files, setFiles] = useState([]);
  const [advOpen, setAdvOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const addFiles = (incoming) => {
    const valid = incoming.filter((f) =>
      SUPPORTED.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setFiles((prev) => [...prev, ...valid]);
  };

  const removeFile = (i) => setFiles((prev) => prev.filter((_, idx) => idx !== i));

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  };

  const depthOptions = [
    { label: "Fast — quick specific questions", value: 3 },
    { label: "Balanced — most questions", value: 5 },
    { label: "Deep — summaries & analysis", value: 10 },
  ];

  return (
    <aside className="panel">
      {/* Logo */}
      <div className="logo">
        <span className="logo-dot" />
        DocMind
      </div>
      <p className="logo-sub">Chat with your documents intelligently</p>

      <hr className="divider" />

      {/* Search Mode */}
      <span className="sec-label">Search Mode</span>
      <select value={searchMode} onChange={(e) => setSearchMode(e.target.value)}>
        <option>Hybrid</option>
        <option>Semantic</option>
        <option>Keyword</option>
      </select>

      <hr className="divider" />

      {/* Advanced */}
      <button className="adv-toggle" onClick={() => setAdvOpen((v) => !v)}>
        Advanced Settings <span>{advOpen ? "▾" : "▸"}</span>
      </button>
      {advOpen && (
        <div className="adv-body">
          <span className="sec-label" style={{ marginBottom: 4 }}>Search Depth</span>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          >
            {depthOptions.map((d) => (
              <option key={d.value} value={d.value}>{d.label}</option>
            ))}
          </select>
        </div>
      )}

      <hr className="divider" />

      {/* Document section */}
      <span className="sec-label">Document</span>

      {!docsLoaded ? (
        <>
          <div
            className={`dropzone${dragging ? " drag" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current.click()}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.docx,.md"
              style={{ display: "none" }}
              onChange={(e) => addFiles(Array.from(e.target.files))}
            />
            <span style={{ fontSize: 24, display: "block", marginBottom: 6 }}>↑</span>
            اسحب أو اختر ملف PDF, DOCX, TXT, MD
          </div>

          {files.map((f, i) => (
            <div className="file-strip" key={i} style={{ marginTop: 6 }}>
              <span className="file-dot" />
              <span className="file-name">{f.name}</span>
              <span
                onClick={() => removeFile(i)}
                style={{ cursor: "pointer", color: "var(--ink3)", marginLeft: 4, fontSize: 12 }}
              >✕</span>
            </div>
          ))}

          {files?.length > 0 && (
            <button
              className="btn-process"
              onClick={() => onProcess(files)}
              disabled={processing}
            >
              {processing ? "Processing…" : "Process Documents"}
            </button>
          )}
        </>
      ) : (
        fileNames.map((name, i) => (
          <div className="file-strip" key={i} style={{ marginBottom: 4 }}>
            <span className="file-dot" />
            <span className="file-name">{name}</span>
            <span className="file-badge">READY</span>
          </div>
        ))
      )}

      {/* Actions */}
      <div className="btn-row">
        <button className="btn btn-light" onClick={onClearChat}>Clear Chat</button>
        <button className="btn btn-dark" onClick={onReset}>Reset All</button>
      </div>
    </aside>
  );
}
