import axios from "axios";
import { API_BASE_URL } from "../utils/constants";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Send a medical question to the backend RAG pipeline.
 * @param {string} question   - The user's medical question.
 * @param {string} session_id - The current chat session ID for memory continuity.
 * @returns {Promise<{answer: string, sources: string[], session_id: string}>}
 */
export const askQuestion = async (question, session_id) => {
  const response = await apiClient.post("/chat", { question, session_id });
  return response.data;
};

/**
 * Clear the chat history for a specific session on the backend.
 * Call this when the user starts a "New Chat" so the AI forgets previous turns.
 * @param {string} session_id - The session to clear.
 * @returns {Promise<{success: boolean, message: string}>}
 */
export const clearSession = async (session_id) => {
  const response = await apiClient.post("/clear", { session_id });
  return response.data;
};

/**
 * Health check to verify the backend is running.
 * @returns {Promise<{status: string}>}
 */
export const healthCheck = async () => {
  const response = await apiClient.get("/health");
  return response.data;
};
