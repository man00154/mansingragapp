import streamlit as st
import os
import requests
import tempfile
import base64
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PDF_DOWNLOAD_DIR = "downloaded_pdfs"
FAISS_INDEX_DIR = "faiss_index"
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)

PREDEFINED_PDF_LINKS = {
    "Dell": [
        "https://i.dell.com/sites/csdocuments/Product_Docs/en/Dell-EMC-PowerEdge-Rack-Servers-Quick-Reference-Guide.pdf",
        "https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-r660xs-technical-guide.pdf",
        "https://i.dell.com/sites/csdocuments/shared-content_data-sheets_documents/en/aa/poweredge_r740_r740xd_technical_guide.pdf",
        "https://dl.dell.com/topicspdf/openmanage-server-administrator-v95_users-guide_en-us.pdf",
        "https://dl.dell.com/manuals/common/dellemc-server-config-profile-refguide.pdf",
    ],
    "IBM": [
        "https://www.redbooks.ibm.com/redbooks/pdfs/sg248513.pdf",
        "https://www.ibm.com/docs/SSLVMB_28.0.0/pdf/IBM_SPSS_Statistics_Server_Administrator_Guide.pdf",
        "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
        "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files",
    ],
    "Cisco": [
        "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
        "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",
    ],
    "Juniper": [
        "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
        "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
        "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",
    ],
    "Fortinet (FortiGate)": [
        "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
        "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
        "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
        "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf",
    ],
    "EUC": [
        "https://www.dell.com/en-us/lp/dt/end-user-computing",
        "https://www.nutanix.com/solutions/end-user-computing",
        "https://eucscore.com/docs/tools.html",
        "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
    ],
}

@st.cache_resource
def get_embeddings():
    # Use a more accurate model for embeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0
    )

def load_and_split_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # Larger chunks + more overlap for full context
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing PDF {file_path}: {e}")
        return []

def load_and_split_webpage(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error processing webpage {url}: {e}")
        return []

def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download PDF: {url}")
    except Exception as e:
        st.error(f"Error downloading PDF from {url}: {e}")
    return False

def fetch_and_process_linked_pdfs(base_url):
    docs = []
    try:
        html = requests.get(base_url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                from urllib.parse import urljoin
                pdf_links.append(urljoin(base_url, href))
        for pdf_url in pdf_links:
            file_name = os.path.basename(urlparse(pdf_url).path)
            output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
            if download_pdf(pdf_url, output_path):
                docs.extend(load_and_split_pdf(output_path))
    except Exception as e:
        st.error(f"Error fetching linked PDFs from {base_url}: {e}")
    return docs

def load_webpage_and_pdfs_parallel(urls):
    all_docs = []
    def process_url(url):
        docs_for_url = []
        docs_for_url.extend(load_and_split_webpage(url))
        docs_for_url.extend(fetch_and_process_linked_pdfs(url))
        return docs_for_url
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                result_docs = future.result()
                all_docs.extend(result_docs)
            except Exception as e:
                st.error(f"Error in parallel processing: {e}")
    return all_docs

def ingest_selected_source(selected_source):
    all_docs = []
    if selected_source in PREDEFINED_PDF_LINKS:
        urls = PREDEFINED_PDF_LINKS[selected_source]
        if selected_source == "EUC":
            all_docs.extend(load_webpage_and_pdfs_parallel(urls))
        else:
            for url in urls:
                if url.endswith(".pdf"):
                    file_name = os.path.basename(urlparse(url).path)
                    output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
                    if download_pdf(url, output_path):
                        all_docs.extend(load_and_split_pdf(output_path))
                else:
                    all_docs.extend(load_and_split_webpage(url))
    return all_docs

def initialize_vector_store(documents, embeddings):
    if documents:
        if "vector_store" in st.session_state and st.session_state.vector_store:
            st.session_state.vector_store.add_documents(documents)
        else:
            st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
        st.session_state.vector_store.save_local(FAISS_INDEX_DIR)

def get_rag_chain(vector_store, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    # Retriever tuned for better recall
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

# ===== Streamlit UI =====
st.set_page_config(layout="wide", page_title="RAG App with Groq")
st.title("📄 MANISH SINGH - RAG Application with Document & Web Chat (Groq, FAISS)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Load models
try:
    embeddings = get_embeddings()
    llm = get_llm()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load local FAISS index
if os.path.exists(FAISS_INDEX_DIR):
    try:
        st.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
    except Exception as e:
        st.warning(f"Failed to load local FAISS index: {e}")

with st.sidebar:
    st.header("Upload & Ingest")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                docs = load_and_split_pdf(tmp.name)
                all_docs.extend(docs)
        initialize_vector_store(all_docs, embeddings)
        st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
        st.session_state.messages = []
        st.success("Uploaded and indexed successfully!")
        st.rerun()

    selected_company = st.selectbox("Predefined Sources", [""] + list(PREDEFINED_PDF_LINKS.keys()))
    if selected_company and st.button("Ingest Selected Source"):
        with st.spinner("Ingesting..."):
            docs = ingest_selected_source(selected_company)
            initialize_vector_store(docs, embeddings)
            st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
            st.session_state.messages = []
            st.success("Documents ingested successfully!")
            st.rerun()

# Chat interface
st.subheader("💬 Chat with Data")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if st.session_state.rag_chain:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({
                        "question": prompt,
                        "chat_history": [
                            (m["role"], m["content"])
                            for m in st.session_state.messages if m["role"] != "assistant"
                        ]
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload or ingest documents first.")
