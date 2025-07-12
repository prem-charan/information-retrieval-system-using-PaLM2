import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    """Handle user input and generate response"""
    if st.session_state.conversation is None:
        st.error("Please upload and process a PDF file first!")
        return
    
    try:
        with st.spinner("Generating response..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write("👤 **User:** ", message.content)
            else:
                st.write("🤖 **Assistant:** ", message.content)
                
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        print(f"Full error: {e}")

def main():
    st.set_page_config(
        page_title="PDF Information Retrieval System", 
        page_icon="📚",
        layout="wide"
    )
    
    st.header("📚 PDF Information Retrieval System")
    st.markdown("Upload PDF documents and ask questions about their content!")
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = False
    
    # Main chat interface
    user_question = st.text_input("Ask a question about your PDF:", 
                                placeholder="e.g., What is the main topic of the document?")
    
    if user_question:
        user_input(user_question)
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("📄 Document Upload")
        st.markdown("Upload your PDF files to get started")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files", 
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files at once"
        )
        
        if st.button("Process Documents", type="primary"):
            if pdf_docs:
                try:
                    with st.spinner("Processing documents... This may take a few minutes."):
                        # Step 1: Extract text from PDFs
                        st.info("📖 Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if raw_text is None:
                            st.error("❌ Failed to extract text from PDFs")
                            return
                        
                        # Step 2: Create text chunks
                        st.info("✂️ Creating text chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                            st.error("❌ Failed to create text chunks")
                            return
                        
                        # Step 3: Create vector store
                        st.info("🔍 Creating vector embeddings...")
                        vector_store = get_vector_store(text_chunks)
                        
                        if vector_store is None:
                            st.error("❌ Failed to create vector store")
                            return
                        
                        # Step 4: Create conversation chain
                        st.info("🤖 Setting up local AI system...")
                        conversation_chain = get_conversational_chain(vector_store, text_chunks)
                        
                        if conversation_chain is None:
                            st.error("❌ Failed to create conversation chain")
                            return
                        
                        # Success!
                        st.session_state.conversation = conversation_chain
                        st.session_state.processed_docs = True
                        st.success("✅ Documents processed successfully! You can now ask questions.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    print(f"Full error: {e}")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        # Display processing status
        if st.session_state.processed_docs:
            st.success("✅ Documents ready for questions!")
        else:
            st.info("📝 Upload and process documents to get started")
        
        # Instructions
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload one or more PDF files
        2. Click 'Process Documents'
        3. Wait for processing to complete
        4. Ask questions about your documents
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        st.markdown("""
        - **System**: Local AI (No API Required)
        - **Embeddings**: Sentence Transformers (Local)
        - **Search**: Semantic + Keyword Search
        - **Vector Store**: FAISS (Local)
        - **Cost**: Completely Free
        - **Status**: 100% Offline
        """)

if __name__ == "__main__":
    main()
