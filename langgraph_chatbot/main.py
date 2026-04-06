import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── LLM Setup ──────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ── Graph Node ─────────────────────────────────────────────────────────────
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ── Build Graph ────────────────────────────────────────────────────────────
graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)

app = graph.compile()

# ── Conversation Memory ────────────────────────────────────────────────────
conversation_history = []

def chat(user_input: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    conversation_history.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": conversation_history})

    # Extract the last AI message
    ai_message = result["messages"][-1]
    conversation_history.append(ai_message)

    return ai_message.content

def reset_conversation():
    """Clear conversation history to start fresh."""
    global conversation_history
    conversation_history = []
    print("🔄 Conversation reset.\n")

# ── Interactive CLI Chat Loop ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("        🤖  LangGraph + Gemini Chatbot")
    print("=" * 55)
    print("Commands:  'quit' or 'exit' → stop")
    print("           'reset'          → clear history")
    print("           'history'        → show chat log")
    print("-" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == "reset":
            reset_conversation()
            continue

        if user_input.lower() == "history":
            if not conversation_history:
                print("  (No messages yet)")
            for msg in conversation_history:
                role = "You" if isinstance(msg, HumanMessage) else "Bot"
                print(f"  [{role}]: {msg.content}")
            continue

        print("\nBot: ", end="", flush=True)
        try:
            response = chat(user_input)
            print(response)
        except Exception as e:
            print(f"⚠️  Error: {e}")

if __name__ == "__main__":
    main()
