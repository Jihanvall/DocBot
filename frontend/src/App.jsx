import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const METHODS = [
  { value: "tfidf", label: "Keywords", sub: "TF-IDF", hint: "Matches your words. Instant." },
  { value: "embedding", label: "Meaning", sub: "Embeddings", hint: "Understands reworded questions. Slower to index." },
  { value: "hybrid", label: "Best of both", sub: "Hybrid", hint: "Combines both. Slower to index." },
];

function Svg({ children, size = 20, stroke = 2 }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={stroke}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const PlusIcon = () => (
  <Svg size={22} stroke={2.4}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);
const SendIcon = () => (
  <Svg size={20}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </Svg>
);
const CloseIcon = () => (
  <Svg size={14} stroke={2.4}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Svg>
);
const DocIcon = () => (
  <Svg size={18}>
    <path d="M6 3h9l4 4v14H6z" />
    <path d="M14 3v5h5M9 13h7M9 17h5" />
  </Svg>
);
const ChevronIcon = () => (
  <Svg size={14} stroke={2.4}>
    <path d="M6 9l6 6 6-6" />
  </Svg>
);

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function readError(response) {
  try {
    const data = await response.json();
    return typeof data.detail === "string" ? data.detail : "Something went wrong.";
  } catch {
    return "Something went wrong.";
  }
}

function Answer({ text }) {
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  return (
    <p className="answer">
      {parts.map((part, i) =>
        /^\[\d/.test(part) ? (
          <span key={i} className="cite">
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </p>
  );
}

function Typing() {
  return (
    <div className="typing" aria-label="Thinking">
      <span />
      <span />
      <span />
    </div>
  );
}

function Indexing({ slow }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="indexing" role="status">
      <div className="progress">
        <span />
      </div>
      <p>
        Reading and indexing your document... {seconds}s
        {slow && seconds > 20 ? " (embeddings take longer, please wait)" : ""}
      </p>
    </div>
  );
}

function Sources({ passages }) {
  const maxScore = Math.max(1e-9, ...passages.map((p) => p.score));
  return (
    <details className="sources">
      <summary>
        Sources <b>{passages.length}</b>
      </summary>
      <ul>
        {passages.map((p, i) => (
          <li key={`${p.source}-${p.position}`}>
            <details>
              <summary>
                <span className="badge">{i + 1}</span>
                <span className="src">
                  {p.source}, part {p.position}
                </span>
                <span className="bar" title={`score ${p.score}`}>
                  <span style={{ width: `${Math.max(8, (p.score / maxScore) * 100)}%` }} />
                </span>
              </summary>
              <p>{p.text}</p>
            </details>
          </li>
        ))}
      </ul>
    </details>
  );
}

function BotAnswer({ message }) {
  const [copied, setCopied] = useState(false);

  if (message.loading) return <Typing />;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <>
      {message.rewritten && (
        <p className="rewritten">Searched for: {message.rewritten}</p>
      )}
      {message.refused ? (
        <p className="answer refused-text">
          I could not find this in the document, so I will not guess. Try rewording the question or
          switch the search to "Meaning".
        </p>
      ) : (
        <Answer text={message.text} />
      )}
      {!message.refused && (
        <button className="ghost" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      )}
      {message.passages?.length > 0 && <Sources passages={message.passages} />}
    </>
  );
}

function buildHistory(messages) {
  const lastReady = messages.map((m) => m.kind).lastIndexOf("ready");
  return messages
  .slice(lastReady + 1)
  .map((m) => {
    if (m.role === "user" && m.kind === "text") return { role: "user", text: m.text };
    if (m.role === "bot" && m.kind === "answer" && !m.loading && m.text) {
      return {
        role: "assistant",
        text: m.refused ? "I could not find this in the document." : m.text.slice(0, 500),
      };
    }
    return null;
  })
  .filter(Boolean)
  .slice(-6)
}

function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [doc, setDoc] = useState(null);
  const [method, setMethod] = useState("tfidf");
  const [menuOpen, setMenuOpen] = useState(false);
  const [input, setInput] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);
  const endRef = useRef(null);
  const idRef = useRef(0);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const push = (message) => {
    const id = ++idRef.current;
    setMessages((prev) => [...prev, { id, ...message }]);
    return id;
  };

  const patch = (id, changes) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...changes } : m)));

  const uploadFile = async (file) => {
    if (!file || indexing || thinking) return;
    const slow = method !== "tfidf";
    push({ role: "user", kind: "file", name: file.name, size: file.size });
    const waitId = push({ role: "bot", kind: "indexing", slow });
    setIndexing(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("method", method);
      const response = await fetch(`${API_URL}/documents`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      setSessionId(data.session_id);
      setDoc(data);
      const used = METHODS.find((m) => m.value === data.method);
      patch(waitId, {
        kind: "ready",
        text: `${data.source} is ready: ${data.chunks} parts, searched by ${used?.label.toLowerCase()}. Ask me anything about it.`,
      });
    } catch (e) {
      patch(waitId, { kind: "error", text: e.message });
    } finally {
      setIndexing(false);
    }
  };

  const send = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || !sessionId || thinking || indexing) return;
    const history = buildHistory(messages);
    setInput("");
    push({ role: "user", kind: "text", text: question });
    const botId = push({ role: "bot", kind: "answer", loading: true });
    setThinking(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("question", question);
      form.append("history", JSON.stringify(history));
      const response = await fetch(`${API_URL}/ask`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      patch(botId, {
        loading: false,
        text: data.answer,
        refused: data.refused,
        passages: data.passages,
        rewritten: data.standalone_question !== question ? data.standalone_question : null,
      });
    } catch (err) {
      patch(botId, { kind: "error", loading: false, text: err.message });
    } finally {
      setThinking(false);
    }
  };

  const removeDoc = () => {
    setSessionId(null);
    setDoc(null);
  };

  const canType = Boolean(sessionId) && !indexing;
  const current = METHODS.find((m) => m.value === method);

  return (
    <div
      className="page"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        uploadFile(e.dataTransfer.files[0]);
      }}
    >
      <div className="bg" aria-hidden="true">
        <span className="blob b1" />
        <span className="blob b2" />
        <span className="blob b3" />
      </div>

      {dragging && (
        <div className="drop-overlay">
          <div>
            <PlusIcon />
            <p>Drop your document to chat with it</p>
          </div>
        </div>
      )}

      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <span className="brand-logo">
              <DocIcon />
            </span>
            <div>
              <h1>DocBot</h1>
              <small>Chat with your documents</small>
            </div>
          </div>

          <div className="method-wrap">
            <button className="method-pill" onClick={() => setMenuOpen((o) => !o)}>
              Search: <b>{current.label}</b>
              <ChevronIcon />
            </button>
            {menuOpen && (
              <>
                <div className="menu-backdrop" onClick={() => setMenuOpen(false)} />
                <div className="menu" role="menu">
                  {METHODS.map((m) => (
                    <button
                      key={m.value}
                      role="menuitemradio"
                      aria-checked={method === m.value}
                      className={method === m.value ? "active" : ""}
                      onClick={() => {
                        setMethod(m.value);
                        setMenuOpen(false);
                      }}
                    >
                      <strong>
                        {m.label} <small>{m.sub}</small>
                      </strong>
                      <span>{m.hint}</span>
                    </button>
                  ))}
                  <p>Applies to the next document you attach.</p>
                </div>
              </>
            )}
          </div>
        </header>

        <main className="chat">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-logo">
                <DocIcon />
              </div>
              <h2>Chat with any document</h2>
                    <p>Press the + button below to attach a text, Markdown or PDF file, then ask questions.
                     Every answer shows where it came from.</p>
             
              <div className="features">
                <span>Cites its sources</span>
                <span>Says "I don't know"</span>
                <span>Keywords or meaning</span>
              </div>
            </div>
          ) : (
            messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="msg user">
                  <div className="bubble user-bubble">
                    {m.kind === "file" ? (
                      <span className="file-card">
                        <DocIcon />
                        <span>
                          {m.name} <em>{formatBytes(m.size)}</em>
                        </span>
                      </span>
                    ) : (
                      m.text
                    )}
                  </div>
                </div>
              ) : (
                <div key={m.id} className="msg bot">
                  <span className="avatar">
                    <DocIcon />
                  </span>
                  <div className={`bubble bot-bubble ${m.kind === "error" ? "error-bubble" : ""}`}>
                    {m.kind === "indexing" && <Indexing slow={m.slow} />}
                    {m.kind === "ready" && <p className="answer">{m.text}</p>}
                    {m.kind === "error" && <p className="answer">{m.text}</p>}
                    {m.kind === "answer" && <BotAnswer message={m} />}
                  </div>
                </div>
              ),
            )
          )}
          <div ref={endRef} />
        </main>

        <footer className="composer-wrap">
          {doc && (
            <div className="doc-chip">
              <DocIcon />
              <span>
                {doc.source} <em>{doc.chunks} parts</em>
              </span>
              <button onClick={removeDoc} aria-label="Remove the document">
                <CloseIcon />
              </button>
            </div>
          )}
          <form className="composer" onSubmit={send}>
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf"
              hidden
              onChange={(e) => {
                uploadFile(e.target.files[0]);
                e.target.value = "";
              }}
            />
            <button
              type="button"
              className="plus"
              onClick={() => fileRef.current?.click()}
              disabled={indexing || thinking}
              aria-label="Attach a document"
              title="Attach a document"
            >
              <PlusIcon />
            </button>
            <input
              className="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                indexing
                  ? "Indexing your document..."
                  : canType
                    ? "Ask something about your document"
                    : "Attach a document with + to start"
              }
              disabled={!canType}
              aria-label="Your question"
            />
            <button
              type="submit"
              className="send"
              disabled={!canType || !input.trim() || thinking}
              aria-label="Send"
            >
              <SendIcon />
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default App;