import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import pdfplumber
from concurrent.futures import ThreadPoolExecutor

# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def process_single_page(page):
    """Process a single page and return text with metadata"""
    try:
        text = page.extract_text()
        if text and text.strip():
            return {
                'content': text,
                'metadata': {
                    'page': page.page_number,
                    'source': page.pdf_path if hasattr(page, 'pdf_path') else 'unknown'
                }
            }
    except Exception as e:
        st.error(f"Error processing page {page.page_number}: {str(e)}")
    return None

def get_pdf_text(pdf_docs):
    """Extract text and page numbers from PDF documents using parallel processing"""
    text_with_metadata = []
    total_files = len(pdf_docs)
    
    for idx, pdf in enumerate(pdf_docs, 1):
        progress_text = f"Processing file {idx} of {total_files}: {pdf.name}"
        progress_bar = st.progress(0)
        st.text(progress_text)
        
        try:
            with pdfplumber.open(pdf) as pdf_file:
                total_pages = len(pdf_file.pages)
                
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Process pages in parallel
                    futures = []
                    for page in pdf_file.pages:
                        page.pdf_path = pdf.name  # Add pdf name to page object
                        futures.append(executor.submit(process_single_page, page))
                    
                    # Collect results as they complete
                    for i, future in enumerate(futures):
                        progress_bar.progress((i + 1) / total_pages)
                        result = future.result()
                        if result:
                            text_with_metadata.append(result)
                            
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            continue
        finally:
            progress_bar.empty()
    
    return text_with_metadata

def get_text_chunks(text_with_metadata):
    """Split text into chunks while preserving metadata using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size
        chunk_overlap=50,  # Reduced overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Optimized separators
    )
    
    chunks = []
    for item in text_with_metadata:
        split_texts = text_splitter.split_text(item['content'])
        for text in split_texts:
            chunks.append({
                'content': text,
                'metadata': item['metadata']
            })
    return chunks

def get_vectorstore(text_chunks):
    """Create vector store with metadata and batch processing"""
    embeddings = OpenAIEmbeddings()
    texts = [chunk['content'] for chunk in text_chunks]
    metadatas = [chunk['metadata'] for chunk in text_chunks]
    
    # Process in batches
    batch_size = 100
    vectorstore = None
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadata = metadatas[i:i + batch_size]
        
        if vectorstore is None:
            vectorstore = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metadata
            )
        else:
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadata
            )
    
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversation chain with custom prompt"""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')
    
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always mention the page numbers where you found the information.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer: Let me help you with that. Based on the PDF content"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3},  # Retrieve top 3 most relevant chunks
            metadata_field_info={"page": "Page number", "source": "Source document"}
        ),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    try:
        # Add file size handling
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
        for pdf in pdf_docs:
            if pdf.size > MAX_FILE_SIZE:
                st.error(f"File {pdf.name} is too large. Please upload files smaller than 100MB.")
                return False
        
        # Process documents with progress tracking
        with st.spinner("Processing PDFs..."):
            text_with_metadata = get_pdf_text(pdf_docs)
            
        with st.spinner("Chunking text..."):
            text_chunks = get_text_chunks(text_with_metadata)
            
        with st.spinner("Creating vector store..."):
            vectorstore = get_vectorstore(text_chunks)
            
        with st.spinner("Initializing conversation chain..."):
            st.session_state.conversation = get_conversation_chain(vectorstore)
            
        st.session_state.processComplete = True
        return True
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Display initial instructions
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")
