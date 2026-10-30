import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent

load_dotenv()
#1.initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


agent = create_react_agent(
    llm,
    [add_numbers]
)
#4. Create a Streamlit app UI
st.title("My First AI agent")
question = st.text_input("Ask a question:")
if question:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    answer = result["messages"][-1].content
    st.write("AI Agent Response:", answer)