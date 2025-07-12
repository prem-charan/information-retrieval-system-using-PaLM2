# 📚 PDF Information Retrieval System

A powerful local AI system that allows you to upload PDF documents and ask questions about their content. The system uses completely offline models for intelligent text processing and natural language understanding.

## 🌟 Features

- 📄 Upload and process multiple PDF documents simultaneously
- 💬 Interactive chat interface for asking questions about PDF content
- 🧠 Intelligent text processing using local AI models
- 🔍 Advanced semantic search using FAISS vector store
- 💾 Conversation memory to maintain context
- 🎯 Smart document summarization and question answering
- 🆓 Completely free - no API costs or internet required

## 🛠️ Tech Stack

- **Python 3.8**: Core programming language
- **LangChain**: Framework for document processing
- **Streamlit**: Interactive web interface
- **Sentence Transformers**: Local AI embeddings
- **FAISS**: Efficient similarity search and clustering
- **PyPDF2**: PDF processing library

## 🚀 Getting Started

### Prerequisites

- Python 3.8
- Conda (recommended for environment management)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/prem-charan/information-retrieval-system-using-PaLM2.git
   cd information-retrieval-system
   ```

2. **Create and activate conda environment**

   ```bash
   conda create -n llmapp python=3.8 -y
   conda activate llmapp
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit server**

   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## 💡 How to Use

1. Upload one or more PDF documents using the file uploader in the sidebar
2. Click "Process Documents" to analyze the documents
3. Ask questions like:
   - "What is this PDF about?"
   - "Summarize the main points"
   - "What are the key findings?"
4. Get intelligent responses based on your PDF content

## 🔧 System Architecture

- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) - Local
- **Search**: Semantic + Keyword hybrid search
- **Vector Store**: FAISS - Local indexing
- **Processing**: Completely offline - no internet required

## 💰 Cost

This system is completely free to use as it leverages local AI models and runs entirely on your machine with no external API calls.
