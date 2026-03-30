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
 * @param {string} question - The user's medical question.
 * @returns {Promise<{answer: string, sources: string[]}>}
 */
export const askQuestion = async (question) => {
  const response = await apiClient.post("/chat", { question });
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
