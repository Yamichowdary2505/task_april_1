import math
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

python_repl = PythonREPLTool()

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    if b == 0:
        return "Error: Cannot divide by zero."
    return str(a / b)

@tool
def square_root(n: float) -> float:
    """Find the square root of a number."""
    return math.sqrt(n)

@tool
def is_prime(n: int) -> str:
    """Check if a number is prime."""
    if n < 2:
        return f"{n} is NOT prime."
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return f"{n} is NOT prime."
    return f"{n} IS prime ✅"

@tool
def power(base: float, exp: float) -> float:
    """Raise base to the power of exp."""
    return math.pow(base, exp)

tools = [python_repl, add, subtract, multiply, divide, square_root, is_prime, power]

SYSTEM_PROMPT = """You are a smart general-purpose assistant with two superpowers:
1. MATH — you have tools to add, subtract, multiply, divide, find square roots, check prime numbers, and more.
2. PYTHON — you can write and execute real Python code using PythonREPLTool.
Rules:
- For math questions → use the math tools.
- For coding questions → write and run Python code using PythonREPLTool.
- For general knowledge questions → answer directly from your own knowledge.
- Never say you are limited to tools only."""

agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)  # ✅ Fixed

def get_text(message):
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
    return str(message.content)

def chat(query):
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print("Bot:", get_text(result["messages"][-1]))

print("General Agent Ready! Type 'quit' to exit.\n")
while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    chat(user)