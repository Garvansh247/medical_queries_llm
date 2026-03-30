import React, { useRef, useEffect } from "react";
import Message from "./Message";

/**
 * ChatBox displays the list of messages and auto-scrolls to the latest one.
 * @param {{ messages: Array, isLoading: boolean }} props
 */
const ChatBox = ({ messages, isLoading }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-1">
      {messages.map((msg, idx) => (
        <Message
          key={idx}
          role={msg.role}
          content={msg.content}
          sources={msg.sources}
        />
      ))}

      {/* Typing indicator */}
      {isLoading && (
        <div className="flex justify-start mb-4">
          <div className="flex-shrink-0 w-9 h-9 rounded-full bg-teal-600 flex items-center justify-center text-white font-bold text-sm mr-3">
            AI
          </div>
          <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1">
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
};

export default ChatBox;
