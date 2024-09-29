import axios from 'axios';

export const getAIMessage = async (userQuery, chatHistory) => {
  try {
    const history = Array.isArray(chatHistory) ? chatHistory : [];

    // we now send the chat_history in the post request to the backend
    const response = await axios.post('http://127.0.0.1:5000/getAIMessage', {
      query: userQuery,
      chat_history: history,
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching AI message:', error);
    return {
      assistant_response: 'Sorry, there was an error processing your request.',
      chat_history: chatHistory, // Return existing chat history unchanged
    };
  }
};
