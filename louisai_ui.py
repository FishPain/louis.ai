import streamlit as st

st.set_page_config(page_title="Louis.AI - Legal Assistant", layout="wide")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.database import VectorDB, ExtractDocs
from src.model import *
from src.utils import check_required_env_vars
from docx import Document as DocxDocument
from io import BytesIO
import re
import os
import tempfile


# ---------- Setup Resources ----------
@st.cache_resource
def initialize_resources():
    load_dotenv()
    check_required_env_vars()
    db = VectorDB()
    db.enable_hnsw_indexing()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = build_graph()
    app = response.compile()
    return db, model, app


db, model, app = initialize_resources()


# ---------- Functions ----------
def clean_markdown(content):
    content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"\*(.*?)\*", r"\1", content)
    content = re.sub(r"\n-{3,}\n", "\n", content)
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


def save_file_locally(file):
    temp_dir = tempfile.gettempdir()  # You can use a custom directory too

    file_extension = os.path.splitext(file.name)[1]
    temp_file_path = os.path.join(temp_dir, f"uploaded_file{file_extension}")

    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    print(f"File saved at: {temp_file_path}")
    # delete the temp file when done
    return temp_file_path, file.type


def extract_file_content(file_path, file_type):
    # Pass the file path to ExtractDocs instead of the file object
    if file_type == "application/pdf":
        return ExtractDocs().extract_document(file_path, "pdf")
    elif file_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        return ExtractDocs().extract_document(file_path, "docx")
    elif file_type == "text/plain":
        # For text files, you can return raw content instead of calling ExtractDocs
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    return None


def user_wants_file(text):
    keywords = ["generate", "create", "draft", "write", "document", "contract"]
    return any(k in text.lower() for k in keywords)


# ---------- Streamlit Layout ----------
if "document_loading" not in st.session_state:
    st.session_state.document_loading = False

with st.sidebar:
    st.header("Legal Tools")
    tool_option = st.radio(
        "Available Tool:", ["Legal Assistant Chatbot"], key="tool_option"
    )

    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_conversation_index" not in st.session_state:
        st.session_state.current_conversation_index = None
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = "file_uploader_0"
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.conversations.append([])
        st.session_state.current_conversation_index = (
            len(st.session_state.conversations) - 1
        )
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

    current_conversation = st.session_state.conversations[
        st.session_state.current_conversation_index
    ]

    # Show conversation history
    for msg in current_conversation:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---------- FILE UPLOADER ----------
    uploaded_file = st.file_uploader(
        "Upload a file (optional):",
        type=["pdf", "doc", "docx", "txt"],
        key=st.session_state.file_uploader_key,
    )

    if uploaded_file:
        file_path, file_type = save_file_locally(uploaded_file)
        st.session_state["uploaded_file_path"] = file_path
        st.session_state["uploaded_file_type"] = file_type
        st.success(
            "File uploaded and ready to be analyzed after you submit a question."
        )

    user_query = st.chat_input("Enter your legal question...")

    if user_query:
        current_conversation.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        file_path = st.session_state.get("uploaded_file_path", None)
        file_type = st.session_state.get("uploaded_file_type", None)
        file_content = None
        if file_path and file_type:
            with st.spinner("Extracting File Content..."):
                file_content = extract_file_content(file_path, file_type)
                db.add_documents(file_content)
                # Prepare combined query if file exists
                file_content = "\n\n".join(
                    [chunk.page_content for chunk in file_content]
                )

        with st.spinner("Thinking..."):
            # Prepare context
            inputs = {
                "query": user_query,
                "db": db,
                "model": model,
                "vectorstore_summary": "It includes all the trustable legal information available in Australia.",
                "retrieved_docs": [],
                "depth": 0,
                "excluded_file_ids": set(),
                "intent_type": "summarise" if uploaded_file else "qa",
                "user_context": file_content if file_content else "",
            }
            try:
                output = app.invoke(inputs)
                response_data = output.get("response", "")
                ai_message = (
                    response_data["messages"][-1].content
                    if isinstance(response_data, dict) and "messages" in response_data
                    else (
                        response_data.content
                        if hasattr(response_data, "content")
                        else str(response_data)
                    )
                )
            except Exception as e:
                ai_message = f"Error processing query: {str(e)}"

        # Add AI message to conversation and display it
        current_conversation.append({"role": "ai", "content": ai_message})
        st.chat_message("ai").write(ai_message)
        # Clear uploaded file data after processing the question
        st.session_state["uploaded_file_path"] = None
        st.session_state["uploaded_file_type"] = None

        # Optionally, reset file uploader widget by incrementing the key
        current_key = st.session_state.file_uploader_key
        st.session_state.file_uploader_key = (
            f"file_uploader_{int(current_key.split('_')[-1]) + 1}"
        )
        # Generate file if user wants a file
        if user_wants_file(user_query):
            file_buffer, file_name = generate_docx(ai_message)
            st.download_button(
                label="📄 Download Generated Document",
                data=file_buffer,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
