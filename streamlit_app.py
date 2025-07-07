import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from typing import Literal
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
import json
from langchain.schema import AIMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# Clarification output type
class ClarifyOutput(TypedDict):
    type: Literal["question", "summary"]
    result: str
    max_clarify: int

# Embedding and retriever setup
@st.cache_resource
def get_chain():
    model_name = "Hvare/Athena-indobert-finetuned-indonli-SentenceTransformer"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local("notebooks/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key
    )

    # Prompt for main QA
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Anda adalah asisten yang membantu pengguna menemukan produk yang sesuai dengan kebutuhan mereka.
        Berdasarkan konteks berikut, berikan jawaban yang relevan dan informatif.
        Context: {context}
        Pertanyaan: {question}
        Berikan jawaban yang singkat dan jelas, serta jika perlu, rekomendasikan produk yang sesuai dengan kebutuhan pengguna.
        """
    )

    # Prompt for clarification
    clarify_prompt = PromptTemplate(
        input_variables=["question", "chat_history"],
        template="""
Anda adalah asisten yang membantu pengguna menemukan produk yang sesuai dengan kebutuhan mereka.

Tugas Anda adalah mengevaluasi apakah pertanyaan pengguna sudah cukup jelas untuk diberikan rekomendasi produk.

- Jika pertanyaan masih terlalu umum, berikan jenis "question" dan ajukan pertanyaan klarifikasi yang singkat dan relevan.
- Jika pertanyaan sudah cukup jelas, berikan jenis "summary" dan ringkasan singkat dari maksud pertanyaan tersebut.

Jawaban Anda harus dalam format JSON:
{{
  "type": "question" atau "summary",
  "result": "..."
}}

Pertanyaan pengguna: {question}
"""
    )

    # Main QA chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.output_key = "answer"
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # Clarification chain
    clarify_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )
    clarify_chain = LLMChain(
        llm=llm,
        prompt=clarify_prompt,
        output_parser=StrOutputParser(),
        memory=clarify_memory,
        output_key="result"
    )

    return qa_chain, memory, clarify_chain, clarify_memory

def clarify_question(question: str, clarify_chain, clarify_memory) -> dict:
    chat_hist = clarify_memory.load_memory_variables({}).get("chat_history", "")
    result = clarify_chain.invoke({
        "question": question,
        "chat_history": chat_hist
    })

    history_str = ""
    for msg in reversed(result["chat_history"]):
        if isinstance(msg, AIMessage):
            history_str = msg.content.strip()
            break

    # Delete first and last line
    history_cleaned = "\n".join(history_str.splitlines()[1:-1])
    history_json = json.loads(history_cleaned)
    return history_json

def ask_bot(question: str, qa_chain):
    result = qa_chain({"question": question})
    answer = result.get("answer", "Tidak ada jawaban yang ditemukan.")
    docs = result.get("source_documents", [])
    return answer, docs

# Streamlit UI
st.set_page_config(page_title="Chatbot Rekomendasi Produk E-Commerce", page_icon="🛒")
st.title("🛒 Chatbot Rekomendasi Produk E-Commerce")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clarify_mode" not in st.session_state:
    st.session_state.clarify_mode = False
if "clarify_question" not in st.session_state:
    st.session_state.clarify_question = ""
if "clarify_chain" not in st.session_state:
    qa_chain, memory, clarify_chain, clarify_memory = get_chain()
    st.session_state.qa_chain = qa_chain
    st.session_state.memory = memory
    st.session_state.clarify_chain = clarify_chain
    st.session_state.clarify_memory = clarify_memory
if "temp_q" not in st.session_state:
    st.session_state.temp_q = ""

user_input = st.text_input("Tanyakan produk yang Anda butuhkan...", key="input")

if st.button("Kirim") and user_input:
    st.session_state.temp_q += user_input.strip() + " "
    clarify_result = clarify_question(st.session_state.temp_q.strip(), st.session_state.clarify_chain, st.session_state.clarify_memory)
    if clarify_result["type"] == "question":
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", clarify_result["result"]))
        st.session_state.clarify_mode = True
        st.session_state.clarify_question = clarify_result["result"]
    elif clarify_result["type"] == "summary":
        summary_q = clarify_result["result"]
        answer, docs = ask_bot(summary_q, st.session_state.qa_chain)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer, docs))
        st.session_state.clarify_mode = False
        st.session_state.clarify_question = ""
        st.session_state.temp_q = ""  # Reset after answer

# Display chat history
for entry in st.session_state.chat_history:
    if entry[0] == "You":
        st.markdown(f"**You:** {entry[1]}")
    else:
        st.markdown(f"**Bot:** {entry[1]}")
        if len(entry) > 2 and entry[2]:
            st.markdown("**Rekomendasi produk:**")
            
            # Create columns for product cards (3 cards per row)
            cols = st.columns(3)
            for idx, doc in enumerate(entry[2]):
                meta = doc.metadata
                col_idx = idx % 3
                
                with cols[col_idx]:
                    # Product card container
                    with st.container():
                        st.markdown("""
                        <style>
                        .product-card {
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            padding: 16px;
                            margin: 8px 0;
                            background-color: white;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }
                        .product-image {
                            width: 100%;
                            height: 120px;
                            background-color: #f0f0f0;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-bottom: 8px;
                        }
                        .product-title {
                            font-weight: bold;
                            font-size: 14px;
                            margin-bottom: 4px;
                            color: #333;
                        }
                        .product-category {
                            font-size: 12px;
                            color: #666;
                            margin-bottom: 4px;
                        }
                        .product-id {
                            font-size: 11px;
                            color: #999;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Product card HTML
                        product_html = f"""
                        <div class="product-card">
                            <div class="product-image">
                                📦
                            </div>
                            <div class="product-title">{meta.get('name', 'Nama Produk')}</div>
                            <div class="product-category">{meta.get('category', 'Kategori')}{' > ' + meta.get('sub_category', '') if meta.get('sub_category') else ''}</div>
                            <div class="product-id">ID: {meta.get('product_id', 'N/A')}</div>
                        </div>
                        """
                        st.markdown(product_html, unsafe_allow_html=True)
                        
                        # Add some spacing between cards
                        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

# If in clarify mode, show the clarifying question as a prompt
if st.session_state.clarify_mode and st.session_state.clarify_question:
    st.info(f"Bot: {st.session_state.clarify_question}")