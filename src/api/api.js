import axios from 'axios';

export const getAIMessage = async (userQuery) => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/getAIMessage', {
      query: userQuery,
    });
    return response.data.message;
  } catch (error) {
    console.error('Error fetching AI message:', error);
    return {
      role: 'assistant',
      content: 'Sorry, there was an error processing your request.',
    };
  }
};
