# Kerala Panchayat Assistant 🏛️

An elegant AI-powered assistant that helps citizens understand Kerala Panchayat rules, procedures, and services. Built with a beautiful Kerala-themed UI inspired by traditional Kasavu sarees and the state's natural beauty.

![Kerala Panchayat Assistant](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🎨 Elegant Kerala-themed UI** - Inspired by Kasavu sarees with cream, dark green, and gold colors
- **🧠 AI-Powered Responses** - Uses advanced language models to provide clear, simple answers
- **📚 Document Search** - RAG (Retrieval Augmented Generation) system for accurate information
- **⚡ Real-time Status** - Shows device (CUDA/CPU) and response time indicators
- **📱 Responsive Design** - Works seamlessly on desktop and mobile devices
- **🔍 Smart Search** - FAISS-powered vector search for relevant document sections

## 🎯 Design Philosophy

### Elegant Kerala Minimalism
- **Kasavu Inspiration**: Cream background (#FFFDD0) representing the base cloth
- **Natural Green**: Dark green text (#004225) for excellent readability
- **Golden Accents**: Muted gold (#D4AF37) used sparingly for highlights
- **Typography**: Merriweather serif for headings, Inter sans-serif for body text
- **Clean Interface**: Minimalist design without clutter or sample questions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/j22k/kerala-panchayat-assistant.git
   cd kerala-panchayat-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   
   Get your free API key from [Groq Console](https://console.groq.com/)

4. **Prepare the data:**
   ```bash
   # Process PDF documents (run this once)
   python ingest_pdf.py
   ```

5. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with these dependencies:

```txt
--index-url https://download.pytorch.org/whl/cu118

torch
torchvision
torchaudio

groq>=0.28.0
faiss-cpu>=1.7.0  
sentence-transformers>=2.2.0
langchain>=0.1.0
PyPDF2>=3.0.0
streamlit>=1.28.0
numpy>=1.21.0

```

For CUDA support (optional):
```bash
pip install faiss-gpu  # instead of faiss-cpu
```

## 🏗️ Project Structure

```
kerala-panchayat-assistant/
├── app.py                          # Main Streamlit application
├── ingest_pdf.py                   # PDF processing script
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── kerala_panchayat_index.bin     # FAISS index (generated)
├── kerala_chunks.pkl              # Document chunks (generated)
├── documents/                     # PDF documents folder
│   └── kerala_panchayat_act.pdf   # Place your PDFs here
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for language model access

### Customization Options
- **Model Selection**: Change embedding model in `KeralaPanchayatRAG.__init__()`
- **Color Scheme**: Modify CSS variables in the `st.markdown()` section
- **Response Length**: Adjust `max_tokens` in the Groq API call
- **Search Results**: Change `k` parameter in `search_relevant_sections()`

## 🎨 UI Components

### Header
- Clean, centered design with Kerala-inspired styling
- Golden accent border representing Kasavu tradition
- Elegant typography with serif headings

### Chat Interface
- Minimalist input area with placeholder text
- No distracting sample questions on startup
- Responsive design for all screen sizes

### Status Indicators
- **Device Status**: Shows CUDA/CPU in bottom-right corner
- **Response Time**: Displayed after each AI response
- Real-time updates with subtle styling

### Answer Display
- Clean answer containers with golden accent borders
- Expandable source references
- Professional typography for readability

## 🚀 Usage

1. **Start the application** and wait for system initialization
2. **Enter your question** about Kerala Panchayat procedures
3. **Click "Get Answer"** to receive AI-powered guidance
4. **View sources** (expandable) to see referenced documents
5. **Check response time** displayed below each answer

### Example Questions
- "How do I apply for a birth certificate?"
- "What is a Gram Sabha meeting?"
- "How are Panchayat taxes calculated?"
- "What services can I get from my local Panchayat?"

## ⚡ Performance Features

- **CUDA Auto-Detection**: Automatically uses GPU when available
- **Model Caching**: Embedding models cached for faster startup
- **Efficient Search**: FAISS vector database for quick retrieval
- **Response Timing**: Real-time performance monitoring

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS
- **Backend**: Python with FAISS vector database
- **AI Model**: Groq LLaMA for response generation
- **Embeddings**: Sentence Transformers for document search
- **Vector Store**: FAISS for similarity search

### Data Processing
1. PDFs are chunked into searchable sections
2. Text embeddings generated using SentenceTransformers
3. FAISS index created for fast similarity search
4. Chunks and index saved for quick loading

## 🎯 Customization Guide

### Changing Colors
Modify the CSS variables in `app.py`:
```css
:root {
    --cream-bg: #FFFDD0;        /* Background color */
    --dark-green: #004225;      /* Text color */
    --muted-gold: #D4AF37;      /* Accent color */
}
```

### Adding New Documents
1. Place PDF files in the `documents/` folder
2. Run `python ingest_pdf.py` to reprocess
3. Restart the application

### Modifying AI Responses
Edit the system prompt in `answer_query()` method:
```python
system_prompt = """Your custom instructions here..."""
```

## 🐛 Troubleshooting

### Common Issues

**"GROQ_API_KEY not found"**
- Create `.env` file with your API key
- Ensure the file is in the project root directory

**"Data files not found"**
- Run `python ingest_pdf.py` first
- Check that PDF files are in the `documents/` folder

**CUDA not detected**
- Install `faiss-gpu` instead of `faiss-cpu`
- Ensure CUDA drivers are properly installed

**Slow responses**
- Check your internet connection (Groq API)
- Consider using a faster embedding model
- Verify CUDA is being used if available

## 📱 Mobile Responsiveness

The application includes responsive CSS for mobile devices:
- Adjusted padding and font sizes
- Optimized touch targets for buttons
- Smaller status indicators on mobile
- Fluid layout that adapts to screen size

## 🔒 Security Considerations

- API keys stored in environment variables
- No sensitive data logged or cached
- PDF documents processed locally
- HTTPS recommended for production deployment

## 🚀 Deployment

### Local Development
```bash
streamlit run main.py
```

### Production Deployment
Consider using:
- **Streamlit Cloud**: Easy deployment with GitHub integration
- **Docker**: Containerized deployment
- **AWS/GCP/Azure**: Cloud platform deployment
- **Heroku**: Simple web app hosting

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kerala Government** for providing Panchayat documentation
- **Streamlit** for the amazing web framework
- **Groq** for fast language model inference
- **FAISS** for efficient similarity search
- **SentenceTransformers** for high-quality embeddings


---

**Built with ❤️ for the people of Kerala**

*This assistant helps citizens understand Panchayat procedures, but for official matters, always contact your local Panchayat office directly.*