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
if "clarify_count" not in st.session_state:
    st.session_state.clarify_count = 0
if "clarify_chain" not in st.session_state:
    qa_chain, memory, clarify_chain, clarify_memory = get_chain()
    st.session_state.qa_chain = qa_chain
    st.session_state.memory = memory
    st.session_state.clarify_chain = clarify_chain
    st.session_state.clarify_memory = clarify_memory
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

user_input = st.text_input("Tanyakan produk yang Anda butuhkan...", key="input")

if st.button("Kirim") and user_input:
    if st.session_state.clarify_mode:
        clarify_result = clarify_question(user_input, st.session_state.clarify_chain, st.session_state.clarify_memory)
        if clarify_result["type"] == "question":
            st.session_state.clarify_count += 1
            st.session_state.last_summary = clarify_result["result"]
            if st.session_state.clarify_count >= 2:
                # After 2 clarifications, treat as summary and answer
                answer, docs = ask_bot(st.session_state.last_summary, st.session_state.qa_chain)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", answer, docs))
                st.session_state.clarify_mode = False
                st.session_state.clarify_question = ""
                st.session_state.clarify_count = 0
                st.session_state.last_summary = ""
            else:
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
            st.session_state.clarify_count = 0
            st.session_state.last_summary = ""
    else:
        clarify_result = clarify_question(user_input, st.session_state.clarify_chain, st.session_state.clarify_memory)
        if clarify_result["type"] == "question":
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", clarify_result["result"]))
            st.session_state.clarify_mode = True
            st.session_state.clarify_question = clarify_result["result"]
            st.session_state.clarify_count = 1
            st.session_state.last_summary = clarify_result["result"]
        elif clarify_result["type"] == "summary":
            summary_q = clarify_result["result"]
            answer, docs = ask_bot(summary_q, st.session_state.qa_chain)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", answer, docs))
            st.session_state.clarify_mode = False
            st.session_state.clarify_question = ""
            st.session_state.clarify_count = 0
            st.session_state.last_summary = ""

# Display chat history
for entry in st.session_state.chat_history:
    if entry[0] == "You":
        st.markdown(f"**You:** {entry[1]}")
    else:
        st.markdown(f"**Bot:** {entry[1]}")
        if len(entry) > 2 and entry[2]:
            st.markdown("**Rekomendasi produk:**")
            for doc in entry[2]:
                meta = doc.metadata
                st.markdown(
                    f"- **ID:** {meta.get('product_id', '-')}, "
                    f"**Nama:** {meta.get('name', '-')}, "
                    f"**Kategori:** {meta.get('category', '-')}"
                    f"{' > ' + meta.get('sub_category', '-') if meta.get('sub_category') else ''}"
                )
        st.markdown("---")

# If in clarify mode, show the clarifying question as a prompt
if st.session_state.clarify_mode and st.session_state.clarify_question:
    st.info(f"Bot: {st.session_state.clarify_question}")