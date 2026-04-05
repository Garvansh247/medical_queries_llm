import React, { useState, useEffect } from "react";
import ChatBox from "../components/ChatBox";
import { askQuestion, clearSession } from "../services/api";
import {
  APP_NAME,
  APP_TAGLINE,
  PLACEHOLDER_TEXT,
  WELCOME_MESSAGE,
} from "../utils/constants";

/**
 * Generate a unique session ID using the browser's built-in crypto API.
 * This is called once on mount and again whenever the user clicks "New Chat".
 */
const generateSessionId = () => crypto.randomUUID();

const Home = () => {
  // sessionId tracks the current conversation on the backend (for memory).
  // On first load we reuse the ID stored in localStorage so the backend memory
  // is still aligned with what is shown in the chat.
  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem("sessionId");
    if (stored) return stored;
    const newId = generateSessionId();
    localStorage.setItem("sessionId", newId);
    return newId;
  });

  // Restore the previous chat log from localStorage so history survives reloads.
  const [messages, setMessages] = useState(() => {
    try {
      const stored = localStorage.getItem("chatLog");
      if (stored) return JSON.parse(stored);
    } catch {
      // Ignore parse errors and fall back to the default welcome message.
    }
    return [{ role: "assistant", content: WELCOME_MESSAGE, sources: [] }];
  });
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Track whether the component has completed its initial mount so we can skip
  // writing to localStorage on the very first render (the values were either
  // just read from there or already saved during state initialisation).
  const isMounted = React.useRef(false);

  // Keep localStorage in sync whenever the chat log changes.
  useEffect(() => {
    if (!isMounted.current) return;
    localStorage.setItem("chatLog", JSON.stringify(messages));
  }, [messages]);

  // Keep localStorage in sync whenever the session ID changes.
  useEffect(() => {
    if (!isMounted.current) return;
    localStorage.setItem("sessionId", sessionId);
  }, [sessionId]);

  // Mark the component as mounted after the first render so subsequent state
  // changes trigger the localStorage sync effects above.
  useEffect(() => {
    isMounted.current = true;
  }, []);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    // Add user message to the chat log immediately
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      // Pass the current sessionId so the backend can use chat history
      const data = await askQuestion(question, sessionId);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      console.error(err);
      const errMsg =
        err?.response?.data?.detail ||
        "Sorry, I couldn't connect to the server. Please make sure the backend is running.";
      setError(errMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Error: ${errMsg}`,
          sources: [],
        },
      ]);
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
   * 1. Tell the backend to forget the current session's history.
   * 2. Generate a fresh session ID for the next conversation.
   * 3. Reset the local chat log to the welcome message.
   * 4. Clear the persisted chat log and session ID from localStorage.
   */
  const handleNewChat = async () => {
    // Optionally clear the old session on the backend (fire-and-forget)
    try {
      await clearSession(sessionId);
    } catch (e) {
      // Not critical — the backend will just keep an unused history entry
      console.warn("Could not clear session on backend:", e);
    }

    const newId = generateSessionId();
    const freshMessages = [{ role: "assistant", content: WELCOME_MESSAGE, sources: [] }];

    // Persist the new state before updating React so there is no window where
    // a reload could restore the old (now-cleared) chat.
    localStorage.setItem("sessionId", newId);
    localStorage.setItem("chatLog", JSON.stringify(freshMessages));

    // Start fresh locally
    setSessionId(newId);
    setMessages(freshMessages);
    setError(null);
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <header className="bg-teal-700 text-white px-6 py-4 shadow-md flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-bold tracking-wide">🩺 {APP_NAME}</h1>
          <p className="text-teal-200 text-xs mt-0.5">{APP_TAGLINE}</p>
        </div>
        {/* New Chat button: resets session & chat log */}
        <button
          onClick={handleNewChat}
          className="text-xs bg-teal-600 hover:bg-teal-500 text-white px-3 py-1.5 rounded-lg transition-colors"
        >
          New Chat
        </button>
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
          Press <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">Enter</kbd> to send &bull; <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">Shift+Enter</kbd> for new line
        </p>
      </div>
    </div>
  );
};

export default Home;
