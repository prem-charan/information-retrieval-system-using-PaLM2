# 📚 PDF Information Retrieval System with PaLM2

A powerful information retrieval system that allows you to upload multiple PDF documents and ask questions about their content. The system uses Google's PaLM2 model for intelligent text processing and natural language understanding.

## 🌟 Features

- 📄 Upload and process multiple PDF documents simultaneously
- 💬 Interactive chat interface for asking questions about PDF content
- 🧠 Intelligent text processing using Google's PaLM2 model
- 🔍 Advanced semantic search using FAISS vector store
- 💾 Conversation memory to maintain context
- 🎯 Accurate information retrieval from PDF documents

## 🛠️ Tech Stack

- **Python 3.8**: Core programming language
- **LangChain**: Framework for developing applications powered by language models
- **Streamlit**: Interactive web interface
- **PaLM2**: Google's advanced language model for embeddings and text generation
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **PyPDF2**: PDF processing library

## 🚀 Getting Started

### Prerequisites

- Python 3.8
- Conda (recommended for environment management)
- Google API Key for PaLM2

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

4. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key:
   ```ini
   GOOGLE_API_KEY="your-api-key-here"
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
2. Click "Submit & Process" to analyze the documents
3. Type your questions in the text input field
4. View the AI-generated responses based on the PDF content

## 📝 Project Structure

```
information-retrieval-system/
├── src/
│   ├── __init__.py
│   └── helper.py          # Core functionality for PDF processing and AI
├── research/
│   └── trials.ipynb       # Research and testing notebook
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
├── setup.py              # Package configuration
└── .env                  # Environment variables
```
