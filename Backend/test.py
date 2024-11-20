from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# LangChain Model Setup
llm = ChatOllama(
    model="llama2",
    temperature=0,
    # Add other model-specific parameters if needed
)

system_prompt = '''You are an Emotional Support Assistant. Respond with short, empathetic replies (10-15 words) that are designed to reduce the seeker’s emotional distress through supportive, empathetic, and helpful responses. Your goal is to:

1. Identify Emotion Causes: Recognize the underlying causes of the seeker’s emotional distress to understand their feelings better.
2. Understand Emotion Effects: Acknowledge the emotional effects triggered by these causes.
3. Provide Verbal Grooming Strategies: Offer independent and integrated strategies to help the seeker, including information, empathy, and constructive suggestions.
4. Promote Positive Emotional Dynamics: Encourage a healthy emotional exchange by focusing not just on the seeker’s state but also on the emotional dynamics between you and the seeker.

When responding, follow these steps:
Step 1: Understand the seeker’s emotional state and the cause of their distress.
Step 2: Empathize with the seeker’s feelings.
Step 3: Provide constructive, helpful suggestions or guidance, focusing on improving their emotional well-being.
Step 4: Ensure your responses are always compassionate, thoughtful, and supportive.
'''

# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])
chain = prompt | llm

# In-memory storage for chat history
chat_histories = {}

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests. Receives user input and returns the AI's response.
    """
   
    global chat_histories
    # Get user data from request
    data = request.get_json()
    print(data)
    user_id = 1
    user_input = data.get("user_input", "").strip()
    
    
    if not user_id or not user_input:
        return jsonify({"error": "Both 'user_id' and 'user_input' are required."}), 400

    # Get the chat history for this user
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    # Append user message to the chat history
    chat_histories[user_id].append(HumanMessage(content=user_input))
    
    try:
        # Invoke LangChain to generate a response
        
        ai_msg = chain.invoke({
            "messages": chat_histories[user_id]
        })
        
        # Append AI response to the chat history
        chat_histories[user_id].append(AIMessage(content=ai_msg.content))
       
        # Return the response to the user
        return jsonify({"response": ai_msg.content})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    """
    Reset the chat history for a specific user.
    """
    global chat_histories
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "'user_id' is required."}), 400

    if user_id in chat_histories:
        chat_histories[user_id] = []

    return jsonify({"message": f"Chat history for user {user_id} has been reset."})


if __name__ == "__main__":
    app.run(debug=True)
