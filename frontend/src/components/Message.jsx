import React from "react";

/**
 * Renders a single chat message bubble.
 * @param {{ role: "user"|"assistant", content: string, sources?: string[] }} props
 */
const Message = ({ role, content, sources }) => {
  const isUser = role === "user";

  return (
    <div
      className={`flex w-full mb-4 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {/* Avatar */}
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-teal-600 flex items-center justify-center text-white font-bold text-sm mr-3">
          AI
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? "order-first" : ""}`}>
        {/* Bubble */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-teal-600 text-white rounded-br-sm"
              : "bg-gray-100 text-gray-800 rounded-bl-sm"
          }`}
        >
          {content}
        </div>

        {/* Sources */}
        {!isUser && sources && sources.length > 0 && (
          <div className="mt-2 px-1">
            <p className="text-xs font-semibold text-gray-500 mb-1">
              📄 Sources:
            </p>
            <ul className="space-y-0.5">
              {sources.map((src, idx) => (
                <li key={idx} className="text-xs text-teal-700 italic">
                  • {src}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gray-300 flex items-center justify-center text-gray-700 font-bold text-sm ml-3">
          You
        </div>
      )}
    </div>
  );
};

export default Message;
