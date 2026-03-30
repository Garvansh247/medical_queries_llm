import React, { useState } from "react";
import ChatBox from "../components/ChatBox";
import { askQuestion } from "../services/api";
import {
  APP_NAME,
  APP_TAGLINE,
  PLACEHOLDER_TEXT,
  WELCOME_MESSAGE,
} from "../utils/constants";

const Home = () => {
  const [messages, setMessages] = useState([
    { role: "assistant", content: WELCOME_MESSAGE, sources: [] },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const data = await askQuestion(question);
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

  const handleClear = () => {
    setMessages([{ role: "assistant", content: WELCOME_MESSAGE, sources: [] }]);
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
        <button
          onClick={handleClear}
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
