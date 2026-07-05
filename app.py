import streamlit as st
import tempfile
import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Import custom modules
from src.pdf_parser import extract_text_from_pdf, detect_sections
from src.summarizer import PaperSummarizer
from src.ner_extractor import TechnicalNER
from src.rag_qa import DocumentRAG
from src.knowledge_graph import KnowledgeGraphBuilder

st.set_page_config(page_title="AI Research Paper Intelligence", layout="wide")
st.title("📚 AI Research Paper Intelligence System")

# Initialize models only once per session
@st.cache_resource
def load_models():
    summarizer = PaperSummarizer()
    ner = TechnicalNER()
    rag = DocumentRAG()
    kg_builder = KnowledgeGraphBuilder()
    return summarizer, ner, rag, kg_builder

summarizer, ner, rag, kg_builder = load_models()

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file is not None:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(tmp_path)
        sections = detect_sections(full_text)
    
    st.success(f"Extracted {len(full_text)} characters!")
    
    # Initialize RAG in background
    chunks = summarizer.splitter.split_text(full_text)
    rag.ingest_document(chunks)

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Entities & Hyperparameters", "Knowledge Graph", "Q&A (RAG)"])

    with tab1:
        st.header("📄 Paper Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing full document..."):
                summary = summarizer.summarize_long_document(full_text)
                st.write(summary)
        
        st.subheader("Sections Detected")
        st.write(list(sections.keys()))

    with tab2:
        st.header("🔍 Technical Entities (Zero-Shot)")
        if st.button("Extract Entities"):
            with st.spinner("Running GLiNER..."):
                entities = ner.extract_entities(full_text)
                hyperparams = ner.extract_hyperparameters(full_text)
                
                st.subheader("Extracted Hyperparameters")
                st.json(hyperparams)
                
                st.subheader("Key Entities")
                st.json(entities)
                
                # Save in session state for Knowledge graph
                st.session_state['entities'] = entities

    with tab3:
        st.header("🕸️ Knowledge Graph")
        if 'entities' in st.session_state:
            if st.button("Build Graph"):
                with st.spinner("Building Co-occurrence Graph..."):
                    g = kg_builder.build_graph(chunks, st.session_state['entities'])
                    
                    if g.number_of_nodes() > 0:
                        net = Network(height='500px', width='100%', directed=True)
                        net.from_nx(g)
                        net.save_graph("temp_graph.html")
                        
                        HtmlFile = open("temp_graph.html", 'r', encoding='utf-8')
                        source_code = HtmlFile.read() 
                        components.html(source_code, height=550)
                    else:
                        st.info("Not enough entity relationships found in the text.")
        else:
            st.info("Please extract entities in the previous tab first.")

    with tab4:
        st.header("❓ Ask the Paper (RAG)")
        query = st.text_input("Enter your question:")
        if st.button("Ask") and query:
            with st.spinner("Retrieving context and generating answer..."):
                prompt = rag.build_prompt(query)
                answer = summarizer._generate(prompt, max_new_tokens=150)
                st.write("**Answer:**", answer)
                
                with st.expander("View Retrieved Context"):
                    retrieved_chunks = rag.retrieve(query)
                    for i, c in enumerate(retrieved_chunks):
                        st.markdown(f"**Chunk {i+1}:** {c[:300]}...")
                        
    # Clean up temp file
    os.unlink(tmp_path)
