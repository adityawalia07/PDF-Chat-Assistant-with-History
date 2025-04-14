import streamlit as st
# Set page config first - must be the first Streamlit command
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import uuid
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Set up environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Function to verify embeddings model works before proceeding
@st.cache_resource
def load_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Test the embedding with a simple text
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) > 0:
            return embeddings
        else:
            st.error("Embedding model returned empty embeddings during initialization test.")
            return None
    except Exception as e:
        st.error(f"Failed to initialize embeddings model: {str(e)}")
        return None

# Initialize embeddings
embeddings = load_embeddings()

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #2e2e2e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #1e3a5f;
        color: white;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .chat-avatar.user {
        background-color: #4f8bf9;
        color: white;
    }
    .chat-avatar.assistant {
        background-color: #10a37f;
        color: white;
    }
    .chat-content {
        flex-grow: 1;
    }
    .document-pill {
        background-color: #000000;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.title("PDF Chat Assistant")
    
    # Session ID with persistence through session state
    st.subheader("Session Settings")
    
    # Initialize session ID if not present
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        key="session_id_input",
        help="A unique identifier for your chat session. Keep the same ID to continue previous conversations."
    )
    
    # Ensure session ID is saved to session state
    st.session_state.session_id = session_id
    
    # Model selection
    model_options = {
        "Gemma2-9b-It": "Gemma 2 9B",
        "llama3-70b-8192": "Llama 3 70B",
        "mixtral-8x7b-32768": "Mixtral 8x7B"
    }
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Advanced settings
    with st.expander("Advanced Settings", expanded=True):
        chunk_size = st.slider("Chunk Size", 1000, 10000, 5000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 500, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                              help="Higher values make output more random, lower values more deterministic")
        max_tokens = st.slider("Max Response Length", 500, 8000, 3000, 500, 
                              help="Higher values allow for longer responses")
        retrieval_k = st.slider("Number of Retrieved Chunks", 2, 10, 6, 1,
                               help="More chunks provide more context to the model")
        
        # Response length preference
        response_length_options = ["Concise", "Balanced", "Detailed", "Comprehensive"]
        response_style = st.selectbox(
            "Response Style",
            options=response_length_options,
            index=3,  # Default to "Comprehensive"
            help="Controls how detailed the AI's responses will be"
        )
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        if session_id in st.session_state.get("store", {}):
            del st.session_state.store[session_id]
            st.success("Conversation cleared!")
    
    st.divider()
    st.caption("Built with LangChain & Streamlit")

# Main content area
st.title("Chat with your PDFs 📄💬")

# Initialize session state for chat history and other data
if 'store' not in st.session_state:
    st.session_state.store = {}

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Check if embeddings model is available
if embeddings is None:
    st.error("⚠️ Embeddings model failed to initialize. Please check your HF_TOKEN environment variable and network connection.")
    st.stop()

# File uploader section
st.subheader("Upload Documents")
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload PDF documents to chat with",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files to query information from them."
    )

with col2:
    if st.button("Process Documents", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            documents = []
            new_files = []
            
            # Track processed files to avoid reprocessing
            current_files = [f.name for f in uploaded_files]
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    new_files.append(uploaded_file.name)
                    
                    # Use a temporary file to handle the PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    
                    try:
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        
                        # Validate that docs contain actual text
                        valid_docs = []
                        for doc in docs:
                            if doc.page_content and len(doc.page_content.strip()) > 10:
                                valid_docs.append(doc)
                            
                        if valid_docs:
                            documents.extend(valid_docs)
                        else:
                            st.warning(f"No usable text found in {uploaded_file.name}. It might be scanned or contain only images.")
                        
                        # Clean up the temporary file
                        os.unlink(temp_path)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            if documents:
                try:
                    # Split and create embeddings for the documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Validate that splits contain text
                    if not splits:
                        st.error("No text chunks were created from your documents. Please check your files.")
                    else:
                        # Verify each split can be embedded properly
                        test_embedding = embeddings.embed_query(splits[0].page_content)
                        if len(test_embedding) == 0:
                            st.error("Embedding model is returning empty vectors. Please check your HF_TOKEN.")
                        else:
                            # Create or update vectorstore
                            try:
                                if st.session_state.vectorstore is None:
                                    st.session_state.vectorstore = Chroma.from_documents(
                                        documents=splits,
                                        embedding=embeddings,
                                        persist_directory="./chroma_db"
                                    )
                                else:
                                    st.session_state.vectorstore.add_documents(splits)
                                
                                # Update processed files list
                                st.session_state.processed_files.extend(new_files)
                                
                                st.success(f"Successfully processed {len(new_files)} new document(s)!")
                            except ValueError as ve:
                                if "Expected Embeddings to be non-empty" in str(ve):
                                    st.error("Error: Embedding model returned empty vectors. Please check your token and connection.")
                                else:
                                    st.error(f"Vector store error: {str(ve)}")
                            except Exception as e:
                                st.error(f"Failed to create vector store: {str(e)}")
                except Exception as e:
                    st.error(f"Error splitting documents: {str(e)}")
            
            elif not new_files and uploaded_files:
                st.info("All documents have already been processed.")

# Display current processed documents
if st.session_state.processed_files:
    st.write("Processed documents:")
    doc_html = "".join([f'<span class="document-pill">{doc}</span>' for doc in st.session_state.processed_files])
    st.markdown(doc_html, unsafe_allow_html=True)
    
    # Document count and token estimate
    if st.session_state.vectorstore:
        try:
            doc_count = len(st.session_state.vectorstore.get()["ids"])
            st.caption(f"📊 {doc_count} text chunks available for retrieval")
        except Exception as e:
            st.warning(f"Could not retrieve document count: {str(e)}")
else:
    st.info("No documents processed yet. Please upload and process some PDFs to start chatting.")

# Chat interface
st.divider()
st.subheader("Ask Questions About Your Documents")

# Define the get_session_history function
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Get response style directive based on user selection
def get_response_style_directive(style):
    if style == "Concise":
        return "Provide a concise summary of the key points. Be brief but informative."
    elif style == "Balanced":
        return "Provide a balanced response with moderate detail. Cover main points thoroughly."
    elif style == "Detailed":
        return "Provide a detailed response that explores the topic in depth. Include examples and explanations."
    elif style == "Comprehensive":
        return "Provide an extremely thorough and comprehensive response. Explore all relevant aspects of the topic in detail, including background information, implications, and nuanced analysis. Use examples, comparisons, and detailed explanations. Aim for an educational, in-depth response that leaves no stone unturned."
    else:
        return "Provide a balanced response with appropriate detail."

# Initialize chat interface only if documents are processed
if st.session_state.vectorstore is not None:
    # Check for Groq API key
    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY environment variable not found. Please check your .env file.")
    else:
        # Initialize LLM
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=selected_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Initialize retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": retrieval_k}  # Use the user-defined k value
            )
            
            # Set up the contextualization prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm,
                retriever,
                contextualize_q_prompt
            )
            
            # Get the response style directive
            response_directive = get_response_style_directive(response_style)
            
            # Set up the question answering prompt with enhanced instructions for detailed responses
            system_prompt = (
                "You are a highly knowledgeable AI assistant that answers questions based on the provided PDF documents. "
                "Use the following pieces of retrieved context to answer the user's question. "
                f"{response_directive} "
                "If the information is not found in the documents, politely say you don't have that information. "
                "Always provide comprehensive and accurate answers based on the context. "
                "Use appropriate formatting with headings, subheadings, bullet points, and numbered lists to improve readability. "
                "When appropriate, analyze the information critically, make connections between different pieces of information, "
                "and highlight important insights. "
                "Always strive to provide the most valuable and complete response possible. "
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create the question-answering chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Create conversational RAG chain with message history
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            # Get current session history
            session_history = get_session_history(session_id)
            
            # Display chat messages
            for i, msg in enumerate(session_history.messages):
                role = "user" if msg.type == "human" else "assistant"
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message {role}">
                        <div class="chat-avatar {role}">{"👤" if role == "user" else "🤖"}</div>
                        <div class="chat-content">{msg.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # User input
            user_input = st.chat_input("Ask a question about your documents...", disabled=not groq_api_key)
            
            if user_input:
                # Show user message
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-avatar user">👤</div>
                        <div class="chat-content">{user_input}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get response with spinner
                with st.spinner("Thinking... This may take a moment for detailed responses"):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )
                        
                        # Show assistant response
                        with st.container():
                            st.markdown(f"""
                            <div class="chat-message assistant">
                                <div class="chat-avatar assistant">🤖</div>
                                <div class="chat-content">{response['answer']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show retrieved chunks in an expander (optional)
                        with st.expander("View Retrieved Context", expanded=False):
                            st.write("The assistant used these sections from your documents:")
                            for i, doc in enumerate(response.get("context", [])):
                                st.markdown(f"**Chunk {i+1}**")
                                st.text(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")
                        st.info("Try rephrasing your question or upload different documents.")
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
else:
    st.info("Please upload and process documents before starting the chat.")