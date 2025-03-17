import streamlit as st

st.set_page_config(page_title="Louis.AI - Legal Assistant", layout="wide")

import json
import nest_asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.database import VectorDB
from src.model import *
from src.utils import check_required_env_vars
from langchain.schema import Document
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from io import BytesIO
import re

nest_asyncio.apply()
load_dotenv()
check_required_env_vars()

# ---------- Setup Resources ----------
@st.cache_resource
def initialize_resources():
    db = VectorDB()
    db.enable_hnsw_indexing()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = build_graph()
    app = response.compile()
    return db, model, app

db, model, app = initialize_resources()

# ---------- Functions ----------
def clean_markdown(content):
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    content = re.sub(r'\*(.*?)\*', r'\1', content)
    content = re.sub(r'\n-{3,}\n', '\n', content)
    return content

def generate_docx(content, filename="generated_document.docx"):
    cleaned_content = clean_markdown(content)
    doc = DocxDocument()
    for para in cleaned_content.split("\n\n"):
        doc.add_paragraph(para.strip())
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer, filename

def extract_file_content(file):
    if file.type == "application/pdf":
        return "\n".join([p.extract_text() for p in PdfReader(file).pages if p.extract_text()])
    elif file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return "\n".join([p.text for p in DocxDocument(file).paragraphs])
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    return None

def user_wants_file(text):
    keywords = ["generate", "create", "draft", "write", "document", "contract"]
    return any(k in text.lower() for k in keywords)

# ---------- Streamlit Layout ----------

with st.sidebar:
    st.header("Legal Tools")
    tool_option = st.radio("Available Tool:", ["Legal Assistant Chatbot"], key="tool_option")

    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_conversation_index" not in st.session_state:
        st.session_state.current_conversation_index = None
    if "file_uploader_key" not in st.session_state:  
        st.session_state.file_uploader_key = "file_uploader_0"
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.conversations.append([])
        st.session_state.current_conversation_index = len(st.session_state.conversations) - 1
    for i, convo in enumerate(st.session_state.conversations):
        if i != st.session_state.current_conversation_index and not convo:
            continue
        if st.button(f"Conversation {i+1}", key=f"convo_{i}", use_container_width=True):
            st.session_state.current_conversation_index = i

if tool_option == "Legal Assistant Chatbot":
    st.title("Louis.AI - Your Legal Assistant")
    st.write("### Ask me legal questions or upload files for context!")
    if st.session_state.current_conversation_index is None:
        st.session_state.current_conversation_index = 0
        if not st.session_state.conversations:
            st.session_state.conversations.append([])

    current_conversation = st.session_state.conversations[st.session_state.current_conversation_index]

    # Show history
    for msg in current_conversation:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---------- FILE UPLOADER ----------
    uploaded_file = st.file_uploader(
    "Upload a file (optional):",
    type=["pdf", "doc", "docx", "txt"],
    key=st.session_state.file_uploader_key  
    )
    if uploaded_file:
        file_content = extract_file_content(uploaded_file)
        if file_content:
            st.session_state["uploaded_file_content"] = file_content
            st.success("File uploaded and ready to be analyzed after you submit a question.")

    # ---------- USER INPUT ----------
    user_query = st.chat_input("Enter your legal question...")

    if user_query:
        current_conversation.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Prepare combined query if file exists
        combined_query = user_query
        if "uploaded_file_content" in st.session_state:
            combined_query += "\n\nContext from uploaded file:\n" + st.session_state["uploaded_file_content"]

        # Prepare context
        context = [Document(page_content=msg["content"], metadata={"role": msg["role"]}) for msg in current_conversation]

        with st.spinner("Thinking..."):
            inputs = {
                "query": combined_query,
                "db": db,
                "model": model,
                "vectorstore_summary": "It includes reliable information of the constitution and legal documents of Singapore.",
                "retrieved_docs": context,
                "depth": 0,
                "excluded_file_ids": set(),
            }
            try:
                output = app.invoke(inputs)
                response_data = output.get("response", "")
                ai_message = (
                    response_data["messages"][-1].content if isinstance(response_data, dict) and "messages" in response_data
                    else response_data.content if hasattr(response_data, "content")
                    else str(response_data)
                )
            except Exception as e:
                ai_message = f"Error processing query: {str(e)}"

        # Add AI message
        current_conversation.append({"role": "ai", "content": ai_message})
        st.chat_message("ai").write(ai_message)

        # Generate file if user wants a file
        if user_wants_file(user_query):
            file_buffer, file_name = generate_docx(ai_message)
            st.download_button(
                label="📄 Download Generated Document",
                data=file_buffer,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        # Automatically remove file content after displaying response
        if "uploaded_file_content" in st.session_state:
            del st.session_state["uploaded_file_content"]
            st.session_state.file_uploader_key = f"file_uploader_{int(st.session_state.file_uploader_key.split('_')[-1]) + 1}"