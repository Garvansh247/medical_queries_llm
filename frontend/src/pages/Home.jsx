import React, { useState, useEffect, useRef } from "react";
import ChatBox from "../components/ChatBox";
import { askQuestion, clearSession } from "../services/api";
import {
  APP_NAME,
  APP_TAGLINE,
  PLACEHOLDER_TEXT,
  WELCOME_MESSAGE,
} from "../utils/constants";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const generateSessionId = () => crypto.randomUUID();

const createNewSession = () => ({
  id: generateSessionId(),
  title: "New Chat",
  messages: [{ role: "assistant", content: WELCOME_MESSAGE, sources: [] }],
});

const SESSIONS_KEY = "chatSessions";
const CURRENT_SESSION_KEY = "currentSessionId";
const MAX_SESSION_TITLE_LENGTH = 30;

/**
 * Read sessions and the active session ID from localStorage in a single pass.
 * Both pieces of state are derived together so there is no risk of a mismatch
 * (e.g. a freshly-generated session ID that doesn't exist in the sessions array).
 */
const getInitialState = () => {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const savedId = localStorage.getItem(CURRENT_SESSION_KEY);
        const currentSessionId =
          savedId && parsed.some((s) => s.id === savedId)
            ? savedId
            : parsed[0].id;
        return { sessions: parsed, currentSessionId };
      }
    }
  } catch {
    // Ignore parse errors and fall back to a fresh session.
  }
  const newSession = createNewSession();
  return { sessions: [newSession], currentSessionId: newSession.id };
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const Home = () => {
  // Single combined state so sessions and currentSessionId are always initialised
  // from the same localStorage snapshot — no risk of a UUID mismatch on first render.
  const [{ sessions, currentSessionId }, setState] = useState(getInitialState);

  // Convenience updaters that mirror the original two-state API used throughout
  // the rest of the component, keeping the functional-update pattern intact.
  const setSessions = (updater) =>
    setState((prev) => ({
      ...prev,
      sessions:
        typeof updater === "function" ? updater(prev.sessions) : updater,
    }));

  const setCurrentSessionId = (id) =>
    setState((prev) => ({ ...prev, currentSessionId: id }));

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Derived: the active session object and its messages.
  // Falls back to the first available session if currentSessionId is stale.
  const currentSession =
    (sessions.length > 0 &&
      (sessions.find((s) => s.id === currentSessionId) ?? sessions[0])) ||
    null;
  const messages = currentSession?.messages ?? [];

  // Skip localStorage writes on the very first render (values were just read
  // from there or initialised during state setup above).
  const isMounted = useRef(false);

  // Persist both sessions and the active session ID whenever either changes.
  useEffect(() => {
    if (!isMounted.current) return;
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
  }, [sessions, currentSessionId]);

  useEffect(() => {
    isMounted.current = true;
  }, []);

  // ---------------------------------------------------------------------------
  // Event handlers
  // ---------------------------------------------------------------------------

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading || !currentSession) return;

    // Capture the active session ID before any async work.
    const sessionId = currentSession.id;

    // When the user sends their very first message, use it as the chat title.
    const isFirstUserMessage = !currentSession.messages.some(
      (m) => m.role === "user"
    );

    // Append the user message (and optionally update the title) immediately.
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              title: isFirstUserMessage
                ? question.slice(0, MAX_SESSION_TITLE_LENGTH)
                : s.title,
              messages: [...s.messages, { role: "user", content: question }],
            }
          : s
      )
    );

    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const data = await askQuestion(question, sessionId);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId
            ? {
                ...s,
                messages: [
                  ...s.messages,
                  {
                    role: "assistant",
                    content: data.answer,
                    sources: data.sources || [],
                  },
                ],
              }
            : s
        )
      );
    } catch (err) {
      console.error(err);
      const errMsg =
        err?.response?.data?.detail ||
        "Sorry, I couldn't connect to the server. Please make sure the backend is running.";
      setError(errMsg);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId
            ? {
                ...s,
                messages: [
                  ...s.messages,
                  {
                    role: "assistant",
                    content: `⚠️ Error: ${errMsg}`,
                    sources: [],
                  },
                ],
              }
            : s
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /**
   * "New Chat" handler:
   * 1. Optionally tell the backend to forget the current session's history.
   * 2. Create a fresh session object, prepend it to the sessions array, and
   *    make it the active session.
   */
  const handleNewChat = async () => {
    if (currentSession) {
      try {
        await clearSession(currentSession.id);
      } catch (e) {
        // Not critical — the backend will just keep an unused history entry.
        console.warn("Could not clear session on backend:", e);
      }
    }

    const newSession = createNewSession();
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setError(null);
  };

  const handleSelectSession = (id) => {
    setCurrentSessionId(id);
    setError(null);
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* ------------------------------------------------------------------ */}
      {/* Sidebar                                                             */}
      {/* ------------------------------------------------------------------ */}
      <aside className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0">
        {/* New Chat button */}
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full text-sm bg-teal-600 hover:bg-teal-500 text-white px-3 py-2 rounded-lg transition-colors flex items-center gap-2"
          >
            <span className="text-lg leading-none">+</span>
            New Chat
          </button>
        </div>

        <div className="px-3 pb-1 text-xs text-gray-400 uppercase tracking-wider font-semibold">
          Recent
        </div>

        {/* Session list */}
        <nav className="flex-1 overflow-y-auto px-2 pb-4 space-y-0.5">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => handleSelectSession(session.id)}
              title={session.title}
              className={`w-full text-left text-sm px-3 py-2 rounded-lg truncate transition-colors ${
                session.id === currentSessionId
                  ? "bg-gray-700 text-white"
                  : "text-gray-300 hover:bg-gray-800 hover:text-white"
              }`}
            >
              💬 {session.title}
            </button>
          ))}
        </nav>

        {/* App name footer */}
        <div className="p-3 border-t border-gray-700 text-xs text-gray-500">
          🩺 {APP_NAME}
        </div>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* Main content area                                                   */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="bg-teal-700 text-white px-6 py-4 shadow-md flex items-center flex-shrink-0">
          <div>
            <h1 className="text-xl font-bold tracking-wide">🩺 {APP_NAME}</h1>
            <p className="text-teal-200 text-xs mt-0.5">{APP_TAGLINE}</p>
          </div>
        </header>

        {/* Disclaimer banner */}
        <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 text-xs text-amber-700 flex-shrink-0">
          ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes
          only and does not constitute medical advice. Always consult a qualified
          healthcare professional.
        </div>

        {/* Chat area */}
        <ChatBox messages={messages} isLoading={isLoading} />

        {/* Error bar */}
        {error && (
          <div className="px-6 py-2 bg-red-50 border-t border-red-200 text-xs text-red-600 flex-shrink-0">
            {error}
          </div>
        )}

        {/* Input area */}
        <div className="border-t border-gray-200 px-4 py-4 bg-white flex-shrink-0">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <textarea
              className="flex-1 resize-none rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent px-4 py-3 text-sm text-gray-800 placeholder-gray-400 max-h-36 min-h-[48px]"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={PLACEHOLDER_TEXT}
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-5 py-3 rounded-xl text-sm font-semibold transition-colors flex-shrink-0"
            >
              {isLoading ? "..." : "Send"}
            </button>
          </div>
          <p className="text-center text-xs text-gray-400 mt-2">
            Press{" "}
            <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">
              Enter
            </kbd>{" "}
            to send &bull;{" "}
            <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">
              Shift+Enter
            </kbd>{" "}
            for new line
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;
