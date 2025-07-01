import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Model and vectorstore setup
@st.cache_resource
def get_chain():
    model_name = "Hvare/Athena-indobert-finetuned-indonli-SentenceTransformer"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local("notebooks/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=1.3,
        max_tokens=512,
        max_retries=2,
        api_key=api_key
    )

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

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.output_key = "answer"
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

qa_chain = get_chain()

def ask_bot(question: str):
    result = qa_chain({"question": question})
    answer = result.get("answer", "Tidak ada jawaban yang ditemukan.")
    docs = result.get("source_documents", [])
    return answer, docs

# Streamlit UI
st.set_page_config(page_title="Chatbot Rekomendasi Produk E-Commerce", page_icon="🛒")
st.title("🛒 Chatbot Rekomendasi Produk E-Commerce")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Tanyakan produk yang Anda butuhkan...", key="input")

if st.button("Kirim") and user_input:
    answer, docs = ask_bot(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer, docs))

# Display chat history
for entry in st.session_state.chat_history:
    if entry[0] == "You":
        st.markdown(f"**You:** {entry[1]}")
    else:
        st.markdown(f"**Bot:** {entry[1]}")
        if entry[2]:
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