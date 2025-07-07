# Chatbot Rekomendasi Produk E-Commerce

A Streamlit-based chatbot application that helps users find products based on their needs using natural language queries. The chatbot leverages a fine-tuned Indonesian BERT model for embeddings and Google Gemini (Generative AI) for generating responses and clarification.

## Features

- 🤖 Interactive chat interface with Streamlit
- 🔍 Semantic search using FAISS vector database
- 🇮🇩 Indonesian language support
- 📦 Product recommendations based on user queries
- 💬 Clarification step: the bot will ask up to 2 clarifying questions if your query is too vague, then proceed to answer
- 💾 Conversation memory for contextual responses
- 🎯 Fine-tuned embedding model for better Indonesian text understanding

## Prerequisites

- Python 3.8 or higher
- Git
- Google Generative AI (Gemini) API key

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
   
   Add your Google Gemini API key to the `.env` file:
   ```
   GOOGLE_GENAI_API_KEY=your_gemini_api_key_here
   ```
   
   > **Note:** Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

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
2. **Clarification step**: If your question is too vague, the bot will ask up to 2 clarifying questions. After 2 clarifications, it will proceed to answer with the best available information.
3. **Get recommendations**: The chatbot will provide relevant product suggestions
4. **View product details**: Each recommendation shows product ID, name, and category

### Example Queries

- "Saya mencari laptop untuk gaming"
- "Tolong rekomendasikan smartphone dengan kamera bagus"
- "Ada produk fashion wanita yang sedang diskon?"
- "Saya butuh headphone wireless untuk kerja"

## How the Clarification Logic Works

- When you ask a question, the bot first checks if it is clear enough to answer.
- If not, the bot will ask a clarifying question (up to 2 times).
- After 2 clarifications, the bot will use the last clarification as the summary and answer your query.
- This ensures you always get an answer, even if your question is initially vague.

## Configuration

### Model Settings

You can modify the model parameters in `streamlit_app.py`:

```python
# Embedding model
model_name = "Hvare/Athena-indobert-finetuned-indonli-SentenceTransformer"

# LLM settings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

# Retrieval settings
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}  # Number of products to retrieve
)
```

### Customizing the Prompt

Edit the prompt templates in `streamlit_app.py` to change the chatbot's behavior or clarification style.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Error**
   - Ensure your `.env` file exists and contains the correct API key
   - Verify the API key is valid and has sufficient quota

3. **FAISS Index Not Found**
   - Make sure the `faiss_index/` directory exists with `index.faiss` and `index.pkl` files
   - If missing, you may need to rebuild the vector database

4. **Memory Issues**
   - Reduce the `k` parameter in search_kwargs

### Performance Optimization

- **Faster loading**: The embedding model is cached using `@st.cache_resource`
- **Memory efficiency**: Consider using smaller embedding models for production
- **Response time**: Adjust `k` and model parameters for faster responses

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
- `langchain-google-genai`: Gemini LLM integration
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

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the Streamlit documentation: https://docs.streamlit.io/

---

**Happy chatting! 🚀** 