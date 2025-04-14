# 📚 PDF Chat Assistant

A sophisticated conversational AI application that allows you to chat with your PDF documents using state-of-the-art language models and retrieval-augmented generation.

## ✨ Features

- **Interactive PDF Conversations**: Engage in natural conversations about your PDF documents with contextual awareness
- **Multiple LLM Options**: Choose between Gemma, Llama 3, and Mixtral models via Groq's API
- **Session Management**: Persistent chat sessions with unique identifiers for continuing conversations
- **Advanced RAG Implementation**: History-aware retrieval for improved contextual understanding
- **Customizable Settings**: Fine-tune chunk size, overlap, temperature, and response style
- **Modern UI**: Clean, responsive interface with dark mode support and styled chat bubbles

## 🔍 Comparison with Other RAG Applications

This PDF Chat Assistant differs from standard RAG implementations in several key ways:

### Compared to Basic Ollama-Based Chatbots:
- Uses cloud-based LLMs through Groq instead of local Ollama models
- Implements comprehensive conversation history tracking
- Offers advanced document processing with validation and error handling
- Provides configurable response styles (Concise, Balanced, Detailed, Comprehensive)

### Compared to Research Papers RAG:
- Focuses on real-time conversational interactions rather than one-off queries
- Maintains chat history for context-aware responses across multiple questions
- Implements history-aware retrieval for improved question understanding
- Offers more granular control over model parameters and response styles
- Uses HuggingFace embeddings with Chroma vector store instead of FAISS

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Groq API key for accessing language models
- HuggingFace token for embeddings model access
- PDF documents you want to chat with

### Installation



Create a `.env` file in the project root with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Running the Application

```bash
streamlit run app.py
```

Navigate to the URL provided by Streamlit (typically http://localhost:8501) to interact with the application.

## 📋 Usage Guide

1. **Upload Documents**: 
   - Add your PDF files using the file uploader
   - Click "Process Documents" to analyze and index them
   - Wait for processing to complete (indicated by success message)

2. **Configure Settings**:
   - Select your preferred AI model (Gemma, Llama 3, or Mixtral)
   - Adjust advanced settings in the sidebar:
     - Chunk size and overlap for document splitting
     - Temperature for response randomness
     - Max response length
     - Number of retrieved chunks
     - Response style (Concise, Balanced, Detailed, or Comprehensive)

3. **Start Chatting**:
   - Enter questions about your documents in the chat input
   - Receive AI-generated responses based on the document content
   - View retrieved document chunks by expanding "View Retrieved Context"

4. **Manage Sessions**:
   - Use the Session ID to continue conversations at a later time
   - Clear the conversation history with the "Clear Conversation" button

## 🔧 Technical Details

### RAG Pipeline Implementation

The application implements a sophisticated retrieval-augmented generation pipeline:

1. **Document Processing**:
   - PDF loading and validation
   - Text extraction and cleaning
   - Chunking with configurable size and overlap

2. **Vector Embedding**:
   - Uses HuggingFace's all-MiniLM-L6-v2 for document embeddings
   - Persistent Chroma vector store for efficient retrieval

3. **History-Aware Retrieval**:
   - Contextualizes current questions using chat history
   - Reformulates questions to be standalone when needed
   - Retrieves the most relevant document chunks

4. **Response Generation**:
   - Combines retrieved context with conversation history
   - Generates comprehensive, well-formatted responses
   - Adapts detail level based on selected response style

## 📝 Customization

### Modifying Vector Storage

The application uses Chroma as the vector store with persistance enabled:

```python
# To change the persistence directory:
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./your_custom_dir"  # Change this path
)
```

### Adding New Models

To add new LLM options, update the model selection dictionary:

```python
model_options = {
    "Gemma2-9b-It": "Gemma 2 9B",
    "llama3-70b-8192": "Llama 3 70B",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "your-new-model-id": "Your New Model Name"  # Add your model here
}
```

## 🔒 Privacy and Security

- PDF processing happens locally on your machine
- Document text is embedded and stored locally in the Chroma database
- Queries and context are sent to Groq's API for response generation
- Session data persists only during the active Streamlit session

## 🚧 Limitations

- Currently supports PDF files only
- Text-based PDFs work best; scanned documents may not process correctly
- Response quality depends on the selected model and document content quality
- Large documents with many pages may require significant processing time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- Support for additional document formats (DOCX, TXT, etc.)
- Multi-modal capabilities for processing images within PDFs
- Local model support as an alternative to cloud APIs
- Advanced document preprocessing for improved quality
- Document summarization features
- Export chat history to PDF or text formats


## Output


Uploading Chat With pdf with history 2.mp4…


---

Built with ❤️ using Streamlit, LangChain, Chroma, and Groq
