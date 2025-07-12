import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Global model for embeddings (loaded once)
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model once and cache it"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

class LocalQASystem:
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
        self.chat_history = []
        self.embeddings = None
        self.index = None
        self.embedding_model = load_embedding_model()
        
        if self.embedding_model:
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for all text chunks"""
        try:
            print("Creating embeddings for text chunks...")
            self.embeddings = self.embedding_model.encode(self.text_chunks)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            print(f"Created embeddings with dimension {dimension}")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            self.embeddings = None
            self.index = None
    
    def _keyword_search(self, question):
        """Fallback keyword-based search"""
        question_lower = question.lower()
        relevant_chunks = []
        
        # Simple keyword matching
        question_words = re.findall(r'\b\w+\b', question_lower)
        question_words = [word for word in question_words if len(word) > 2]
        
        for i, chunk in enumerate(self.text_chunks):
            chunk_lower = chunk.lower()
            score = 0
            
            # Count keyword matches
            for word in question_words:
                score += chunk_lower.count(word)
            
            if score > 0:
                relevant_chunks.append((i, chunk, score))
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x[2], reverse=True)
        return relevant_chunks[:3]
    
    def _semantic_search(self, question):
        """Semantic search using embeddings"""
        if not self.embedding_model or self.index is None:
            return self._keyword_search(question)
        
        try:
            # Encode the question
            question_embedding = self.embedding_model.encode([question])
            
            # Search in FAISS index
            distances, indices = self.index.search(question_embedding.astype('float32'), 3)
            
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.text_chunks):
                    relevant_chunks.append((idx, self.text_chunks[idx], 1.0 - distances[0][i]))
            
            return relevant_chunks
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return self._keyword_search(question)
    
    def _generate_answer(self, question, relevant_chunks):
        """Generate answer based on relevant chunks"""
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Analyze the question type and provide appropriate response
        question_lower = question.lower()
        
        # Check if it's a general "what is this about" question
        if any(phrase in question_lower for phrase in ['what is this', 'what is the', 'about', 'summary', 'overview']):
            return self._generate_summary(relevant_chunks)
        
        # For specific questions, provide detailed answers
        return self._generate_detailed_answer(question, relevant_chunks)
    
    def _generate_summary(self, relevant_chunks):
        """Generate a summary of the document"""
        # Combine all relevant text
        all_text = " ".join([chunk for _, chunk, _ in relevant_chunks])
        
        # Extract key information patterns
        summary_parts = []
        
        # Check for resume/CV patterns
        if any(keyword in all_text.lower() for keyword in ['experience', 'education', 'skills', 'projects', 'bachelor', 'internship']):
            summary_parts.append("📄 **Document Type**: This appears to be a professional resume/CV")
            
            # Extract name if present
            lines = all_text.split('\n')
            for line in lines[:5]:  # Check first few lines for name
                if len(line.strip()) > 0 and len(line.strip()) < 50 and not any(char.isdigit() for char in line):
                    if line.strip().isupper() or (line.strip().count(' ') <= 3 and line.strip().count(' ') > 0):
                        summary_parts.append(f"👤 **Person**: {line.strip()}")
                        break
            
            # Extract education
            if 'bachelor' in all_text.lower() or 'b.tech' in all_text.lower() or 'engineering' in all_text.lower():
                education_match = re.search(r'(bachelor.*?engineering|b\.tech.*?|computer science.*?engineering)', all_text.lower())
                if education_match:
                    summary_parts.append(f"🎓 **Education**: Computer Science Engineering student")
            
            # Extract skills
            skills_section = re.search(r'skills[:\s]*(.*?)(?:experience|projects|education|$)', all_text.lower(), re.DOTALL)
            if skills_section:
                skills_text = skills_section.group(1)[:200]  # Limit length
                key_skills = []
                for skill in ['python', 'javascript', 'react', 'node.js', 'mongodb', 'html', 'css', 'git']:
                    if skill in skills_text:
                        key_skills.append(skill.title())
                if key_skills:
                    summary_parts.append(f"💻 **Key Skills**: {', '.join(key_skills[:5])}")
            
            # Extract experience
            if 'intern' in all_text.lower() or 'developer' in all_text.lower():
                summary_parts.append("💼 **Experience**: Has internship and development experience")
            
            # Extract projects
            if 'project' in all_text.lower():
                summary_parts.append("🚀 **Projects**: Contains information about technical projects")
        
        # Generic document analysis if not a resume
        elif not summary_parts:
            summary_parts.append("📄 **Document Type**: General document")
            
            # Extract key topics
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(words)
            common_words = [word for word, count in word_freq.most_common(10) 
                          if len(word) > 3 and word not in ['the', 'and', 'with', 'this', 'that', 'have', 'from', 'they', 'been', 'were']]
            
            if common_words:
                summary_parts.append(f"🔍 **Key Topics**: {', '.join(common_words[:5])}")
        
        # Combine summary
        if summary_parts:
            summary = "## Document Summary\n\n" + "\n".join(summary_parts)
            summary += f"\n\n📊 **Document Length**: Approximately {len(all_text.split())} words"
            return summary
        else:
            return "This document contains various information. Please ask a more specific question to get detailed answers."
    
    def _generate_detailed_answer(self, question, relevant_chunks):
        """Generate detailed answer for specific questions"""
        answer = f"Based on the document, here's what I found regarding '{question}':\n\n"
        
        for i, (idx, chunk, score) in enumerate(relevant_chunks, 1):
            # Clean up the chunk
            chunk_clean = chunk.strip()
            if len(chunk_clean) > 400:  # Reduced from 800 for better readability
                chunk_clean = chunk_clean[:400] + "..."
            
            answer += f"**Relevant Information {i}:**\n{chunk_clean}\n\n"
        
        return answer
    
    def answer_question(self, question):
        """Main method to answer questions"""
        try:
            # Get relevant chunks using semantic search (with keyword fallback)
            relevant_chunks = self._semantic_search(question)
            
            # Generate answer
            answer = self._generate_answer(question, relevant_chunks)
            
            return answer
        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}"
    
    def __call__(self, query_dict):
        """Interface compatible with langchain"""
        question = query_dict['question']
        answer = self.answer_question(question)
        
        # Simulate chat history format
        from types import SimpleNamespace
        user_msg = SimpleNamespace(content=question)
        ai_msg = SimpleNamespace(content=answer)
        
        self.chat_history.extend([user_msg, ai_msg])
        
        return {'chat_history': self.chat_history, 'answer': answer}

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            st.error("No text could be extracted from the PDF files. Please check if the PDFs contain readable text.")
            return None
            
        print(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def get_text_chunks(text):
    """Split text into chunks for processing"""
    if not text:
        return []
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        print(f"Created {len(chunks)} text chunks")
        
        if not chunks:
            st.error("No valid text chunks created. Please check your PDF content.")
            return []
            
        return chunks
    except Exception as e:
        st.error(f"Error creating text chunks: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create vector store - not used in local version"""
    # This is kept for compatibility but not used
    return True

def get_conversational_chain(vector_store, text_chunks):
    """Create local QA system"""
    if not text_chunks:
        st.error("No text chunks provided for QA system")
        return None
        
    try:
        qa_system = LocalQASystem(text_chunks)
        print("Local QA system created successfully")
        return qa_system
        
    except Exception as e:
        st.error(f"Error creating QA system: {str(e)}")
        print(f"Full error: {e}")
        return None
