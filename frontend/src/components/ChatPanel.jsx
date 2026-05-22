import { useEffect, useRef, useState } from "react";

function UserBubble({ content }) {
  return (
    <div className="msg-wrap-user">
      <span className="msg-role">You</span>
      <div className="bubble-user">{content}</div>
    </div>
  );
}

function AiBubble({ content, citations }) {
  return (
    <div className="msg-wrap-ai">
      <span className="msg-role">DocMind</span>
      <div className="bubble-ai">
        <div dangerouslySetInnerHTML={{ __html: content }} />
        {citations?.length > 0 && (
          <div className="cit-wrap">
            {citations.map((c, i) => (
              <span className="cit-item" key={i}>
                <span className="cit-src">
                  {c.source}{c.page ? ` · p.${c.page}` : ""}
                </span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="msg-wrap-ai">
      <span className="msg-role">DocMind</span>
      <div className="bubble-ai">
        <div className="thinking">
          <div className="dot" /><div className="dot" /><div className="dot" />
        </div>
      </div>
    </div>
  );
}

export default function ChatPanel({ history, thinking, docsLoaded, fileNames, onSend, error }) {
  const [input, setInput] = useState("");
  const messagesRef = useRef();
  const textareaRef = useRef();

  useEffect(() => {
    if (messagesRef.current)
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [history, thinking]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || thinking || !docsLoaded) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    onSend(q);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const autoResize = (el) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <section className="chat-col">
      <div className="chat-header">
        <div className="chat-title">
          Conversation
          {fileNames?.length > 0 && (
            <span className="chat-doc-tag">{fileNames[0]}</span>
          )}
        </div>
      </div>
      <div className="messages" ref={messagesRef}>
        {!history?.length && !thinking ? (
          <div className="empty-state">
            <span style={{ fontSize: 40, opacity: 0.2 }}>📄</span>
            <span style={{ fontSize: "0.82rem", color: "var(--ink3)" }}>
              {docsLoaded ? "اسأل أي شيء عن مستنداتك" : "ارفع ملفاً لتبدأ المحادثة"}
            </span>
          </div>
        ) : (
          <>
            {history?.map((msg, i) =>
              msg.role === "user"
                ? <UserBubble key={i} content={msg.content} />
                : <AiBubble key={i} content={msg.content} citations={msg.citations} />
            )}
            {thinking && <ThinkingBubble />}
          </>
        )}
      </div>
      {error && <div className="err-box">{error}</div>}
      <div className="chat-input-wrap">
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={(e) => { setInput(e.target.value); autoResize(e.target); }}
          onKeyDown={handleKey}
          placeholder={docsLoaded ? "اسأل أي شيء عن مستنداتك…" : "ارفع مستنداً للبدء…"}
          disabled={!docsLoaded || thinking}
          rows={1}
        />
        <button className="send-btn" onClick={handleSend}
          disabled={!docsLoaded || thinking || !input.trim()} aria-label="إرسال">
          ↗
        </button>
      </div>
    </section>
  );
}
