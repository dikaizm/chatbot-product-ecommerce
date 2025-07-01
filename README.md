# Chatbot Rekomendasi Produk E-Commerce

A Streamlit-based chatbot application that helps users find products based on their needs using natural language queries. The chatbot leverages a fine-tuned Indonesian BERT model for embeddings and DeepSeek LLM for generating responses.

## Features

- 🤖 Interactive chat interface with Streamlit
- 🔍 Semantic search using FAISS vector database
- 🇮🇩 Indonesian language support
- 📦 Product recommendations based on user queries
- 💾 Conversation memory for contextual responses
- 🎯 Fine-tuned embedding model for better Indonesian text understanding

## Prerequisites

- Python 3.8 or higher
- Git
- DeepSeek API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd chatbot-product-ecommerce
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your DeepSeek API key to the `.env` file:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```
   
   > **Note:** Get your DeepSeek API key from [DeepSeek's official website](https://platform.deepseek.com/)

## Project Structure

```
chatbot-product-ecommerce/
├── data/                          # Product data files
│   ├── data_products_id_small.csv
│   ├── data_products_id_tiny.csv
│   └── data_products_id_tiny_with_desc.csv
├── notebooks/                     # Jupyter notebooks for data processing
│   ├── data_cleaning.ipynb
│   ├── data_preprocessing.ipynb
│   └── desc_checkpoints/         # Generated product descriptions
├── faiss_index/                  # FAISS vector database
│   ├── index.faiss
│   └── index.pkl
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### Running the Streamlit App

1. **Activate your virtual environment** (if not already activated)
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the application**
   
   The app will automatically open in your default browser. If it doesn't, navigate to:
   ```
   http://localhost:8501
   ```

### Using the Chatbot

1. **Start a conversation**: Type your product-related questions in Indonesian
2. **Get recommendations**: The chatbot will provide relevant product suggestions
3. **View product details**: Each recommendation shows product ID, name, and category

### Example Queries

- "Saya mencari laptop untuk gaming"
- "Tolong rekomendasikan smartphone dengan kamera bagus"
- "Ada produk fashion wanita yang sedang diskon?"
- "Saya butuh headphone wireless untuk kerja"

## Configuration

### Model Settings

You can modify the model parameters in `streamlit_app.py`:

```python
# Embedding model
model_name = "Hvare/Athena-indobert-finetuned-indonli-SentenceTransformer"

# LLM settings
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1.3,        # Adjust creativity (0.0-2.0)
    max_tokens=512,         # Maximum response length
    max_retries=2,          # Retry attempts
    api_key=api_key
)

# Retrieval settings
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}  # Number of products to retrieve
)
```

### Customizing the Prompt

Edit the prompt template in `streamlit_app.py` to change the chatbot's behavior:

```python
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
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Error**
   - Ensure your `.env` file exists and contains the correct API key
   - Verify the API key is valid and has sufficient credits

3. **FAISS Index Not Found**
   - Make sure the `faiss_index/` directory exists with `index.faiss` and `index.pkl` files
   - If missing, you may need to rebuild the vector database

4. **Memory Issues**
   - Reduce the `k` parameter in search_kwargs
   - Lower the `max_tokens` parameter

### Performance Optimization

- **Faster loading**: The embedding model is cached using `@st.cache_resource`
- **Memory efficiency**: Consider using smaller embedding models for production
- **Response time**: Adjust `max_tokens` and `temperature` for faster responses

## Development

### Adding New Features

1. **Custom UI components**: Modify the Streamlit interface in `streamlit_app.py`
2. **New data sources**: Update the data loading and preprocessing notebooks
3. **Different models**: Replace the embedding or LLM models as needed

### Data Processing

The project includes Jupyter notebooks for data processing:
- `data_cleaning.ipynb`: Clean and prepare product data
- `data_preprocessing.ipynb`: Create embeddings and build FAISS index

## Dependencies

Key dependencies include:
- `streamlit`: Web application framework
- `langchain`: LLM framework
- `langchain-deepseek`: DeepSeek LLM integration
- `sentence-transformers`: HuggingFace embeddings
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment variable management

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Happy chatting! 🚀** 