import os
import google.generativeai as genai
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_utils import init_gemini_model

llm = init_gemini_model()

FILE_PATH = r"C:\Sourcsyes\project\langchain_agent\Complete_Vegan_Meat_Recipe_Guide.pdf"  # Change to your file name (.pdf or .txt)

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")
    return loader.load()

pages    = load_document(FILE_PATH)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(pages)

print(f"Loaded '{FILE_PATH}' → {len(chunks)} chunks ready")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever    = vector_store.as_retriever(search_kwargs={"k": 3})

print("Vector store created successfully!")

@tool
def search_document(query: str) -> str:
    """Search the document and return relevant information for a given query."""
    docs   = retriever.invoke(query)
    result = ""
    for doc in docs:
        result += f"Page {doc.metadata.get('page', '?')}: {doc.page_content}\n\n"
    return result if result else "No relevant information found in the document."

tools = [search_document]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that answers questions by searching through the provided document. Always use the search_document tool to find answers."
)

questions = [
    "What is the main topic of this document?",
    "Summarize the key points from the document.",
    "What are the most important details mentioned?",
    "What ingredients are needed for Seitan Steak?",
    "How long can vegan meat be stored in the refrigerator?",
    "What are the protein sources recommended in this guide?"
]

for question in questions:
    print("\n" + "=" * 50)
    print(f"Question: {question}")
    print("=" * 50)
    result  = agent.invoke({"messages": [{"role": "user", "content": question}]})
    content = result['messages'][-1].content
    if isinstance(content, list):
        answer = " ".join([c['text'] for c in content if c.get('type') == 'text'])
    else:
        answer = content
    print(f"Answer: {answer}")