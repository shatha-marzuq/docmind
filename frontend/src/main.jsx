import { useState } from "react";
import LeftPanel from "./components/LeftPanel";
import ChatPanel from "./components/ChatPanel";
import { uploadFiles, askQuestion, resetAll } from "./api/docmind";

export default function App() {
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [fileNames, setFileNames] = useState([]);
  const [history, setHistory] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [searchMode, setSearchMode] = useState("Hybrid");
  const [topK, setTopK] = useState(5);

  const handleProcess = async (files) => {
    setProcessing(true);
    setError(null);
    try {
      const res = await uploadFiles(files);
      setFileNames(res.file_names);
      setDocsLoaded(true);
      setHistory([]);
    } catch (e) {
      setError(e.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleSend = async (question) => {
    setHistory((h) => [...h, { role: "user", content: question }]);
    setThinking(true);
    setError(null);
    try {
      const res = await askQuestion({ question, searchMode, topK });
      setHistory((h) => [
        ...h,
        { role: "assistant", content: res.answer, citations: res.citations },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setThinking(false);
    }
  };

  const handleClearChat = () => setHistory([]);

  const handleReset = async () => {
    try {
      await resetAll();
    } catch (_) {}
    setDocsLoaded(false);
    setFileNames([]);
    setHistory([]);
    setError(null);
  };

  return (
    <div className="layout">
      <LeftPanel
        docsLoaded={docsLoaded}
        fileNames={fileNames}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        topK={topK}
        setTopK={setTopK}
        onProcess={handleProcess}
        onClearChat={handleClearChat}
        onReset={handleReset}
        processing={processing}
      />
      <ChatPanel
        history={history}
        thinking={thinking}
        docsLoaded={docsLoaded}
        fileNames={fileNames}
        onSend={handleSend}
        error={error}
      />
    </div>
  );
}
