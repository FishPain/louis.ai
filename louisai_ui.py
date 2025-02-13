import streamlit as st

# Streamlit UI
st.set_page_config(page_title="Louis.AI - Legal Assistant", layout="wide")

with st.sidebar:
    st.header("Legal Tools")
    tool_option = st.radio(
        "Choose an option:",
        [
            "Chat",
            "Draft Document",
            "Legal Information Lookup",
            "File Upload"  # This remains as a separate option if needed
        ],
        key="tool_option"
    )
    
    # Sidebar chat history (only for Chat tool)
    if tool_option == "Chat":
        st.header("Chat History")
        
        if "conversations" not in st.session_state:
            st.session_state.conversations = []
        if "current_conversation_index" not in st.session_state:
            st.session_state.current_conversation_index = None
        
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.conversations.append([])  # Start a new empty conversation
            st.session_state.current_conversation_index = len(st.session_state.conversations) - 1
        
        for i, convo in enumerate(st.session_state.conversations):
            if i != st.session_state.current_conversation_index and not convo:
                continue
            if st.button(f"Conversation {i+1}", key=f"convo_{i}", use_container_width=True):
                st.session_state.current_conversation_index = i

if tool_option == "Chat":
    st.title("Louis.AI - Your Legal Assistant")
    st.write("### Ask me legal questions or upload files for context!")
    
    # Ensure valid conversation index
    if st.session_state.current_conversation_index is None:
        st.session_state.current_conversation_index = 0
        if not st.session_state.conversations:
            st.session_state.conversations.append([])
    
    current_conversation = st.session_state.conversations[st.session_state.current_conversation_index]
    
    # Display chat messages
    for msg in current_conversation:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # --------- File Uploader in Chat Area ---------
    # Using a key specific for chat so it doesn't conflict with the standalone file upload option.
    uploaded_file = st.file_uploader(
        "Upload a file (optional):",
        type=["pdf", "doc", "docx", "txt"],
        key="chat_file_uploader"
    )
    
    # Process the uploaded file only once per unique file name
    if uploaded_file is not None:
        # Use a session state variable to ensure we process a new file only once.
        if "last_chat_file" not in st.session_state or st.session_state.last_chat_file != uploaded_file.name:
            st.session_state.last_chat_file = uploaded_file.name
            
            file_details = (
                f"**Uploaded file:** {uploaded_file.name}  \n"
                f"**Type:** {uploaded_file.type}  \n"
                f"**Size:** {uploaded_file.size} bytes"
            )
            # Append file details as a chat message from the user
            current_conversation.append({"role": "user", "content": file_details})
            st.chat_message("user").write(file_details)
            
            # If it's a plain text file, also display its content.
            if uploaded_file.type == "text/plain":
                try:
                    # Using getvalue() to retrieve the entire file content
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    content_message = f"**File Content:**\n{file_content}"
                    current_conversation.append({"role": "user", "content": content_message})
                    st.chat_message("user").write(content_message)
                except Exception as e:
                    error_message = f"Error reading file: {e}"
                    current_conversation.append({"role": "user", "content": error_message})
                    st.chat_message("user").write(error_message)
            else:
                note_message = "Preview not available for this file type."
                current_conversation.append({"role": "user", "content": note_message})
                st.chat_message("user").write(note_message)
    
    # --------- Chat Input Area ---------
    user_query = st.chat_input("Enter your legal question...")
    
    if user_query:
        # Append user query to chat history
        current_conversation.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Generate AI response (Replace this with actual AI logic)
        response = "(Louis.AI Response Placeholder)"
        
        # Append AI response to chat history
        current_conversation.append({"role": "ai", "content": response})
        st.chat_message("ai").write(response)

elif tool_option == "Legal Information Lookup":
    st.title("Legal Information Lookup")
    st.write("### Quickly retrieve relevant legal information!")
    
    search_query = st.text_input("Enter legal topic or keyword:")

    if st.button("Search"):
        st.write("(Legal information retrieval logic goes here)")

elif tool_option == "Draft Document":
    st.title("Draft Legal Documents")
    st.write("### Generate legal documents effortlessly!")
    
    doc_type = st.selectbox("Select document type:", ["Contract", "Will", "Affidavit", "NDA"])
    
    if doc_type == "Contract":
        contract_type = st.selectbox("Type of Contract:", ["Employment", "Service", "Partnership", "Sales", "Lease", "Other"])
        parties_involved = st.text_input("Parties Involved (e.g., Company A & Company B)")
        contract_duration = st.text_input("Contract Duration (e.g., 1 year, 6 months)")
        payment_terms = st.text_area("Payment Terms")
        obligations = st.text_area("Obligations of Parties")
        termination_conditions = st.text_area("Termination Conditions")
        additional_clauses = st.text_area("Additional Clauses")
        
    elif doc_type == "Will":
        testator_name = st.text_input("Testator's Name")
        beneficiaries = st.text_area("Beneficiaries and Their Allocations")
        executor_name = st.text_input("Executor's Name")
        special_instructions = st.text_area("Special Instructions")
        
    elif doc_type == "Affidavit":
        affiant_name = st.text_input("Affiant's Name")
        statement_details = st.text_area("Statement Details")
        witnesses = st.text_area("Names of Witnesses")
        
    elif doc_type == "NDA":
        disclosing_party = st.text_input("Disclosing Party")
        receiving_party = st.text_input("Receiving Party")
        confidentiality_period = st.text_input("Confidentiality Period (e.g., 2 years)")
        exclusions = st.text_area("Exclusions from Confidentiality")
        governing_law = st.text_input("Governing Law")
        contract_type = st.selectbox("Type of Contract:", ["Employment", "Service", "Partnership", "Sales", "Lease", "Other"])
        parties_involved = st.text_input("Parties Involved (e.g., Company A & Company B)")
        contract_duration = st.text_input("Contract Duration (e.g., 1 year, 6 months)")
        payment_terms = st.text_area("Payment Terms")
        obligations = st.text_area("Obligations of Parties")
        termination_conditions = st.text_area("Termination Conditions")
        additional_clauses = st.text_area("Additional Clauses")
        
    client_name = st.text_input("Client Name")
    additional_info = st.text_area("Additional Information", height=150)
    
    if st.button("Generate Document"):
        draft_text = f"Legal Document: {doc_type}\nClient: {client_name}\nDetails: {additional_info}"
        st.text_area("Generated Document:", draft_text, height=400)
        st.download_button("Download Document", draft_text, file_name=f"{doc_type}_draft.txt")

elif tool_option == "File Upload":
    st.title("Upload Document")
    st.write("### Drag and drop your file or click to select one.")
    
    # File uploader supports drag and drop by default
    uploaded_file = st.file_uploader(
        "Upload your file:",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        st.write("**File Details:**")
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size)
        
        # Example: Display text file contents (if applicable)
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", file_content, height=400)
        else:
            st.info("Preview not available for this file type.")
