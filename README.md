# ATQ - Any Time Questions

## Project Description

ATQ (Any Time Questions) is a Retrieval-Augmented Generation (RAG) based chatbot that enables users to interact with website content through natural language questions. The application scrapes website data, converts it into embeddings, stores it in a vector database, and generates answers using a Large Language Model.

## Features

* Website content scraping
* Text chunking
* Embedding generation using Sentence Transformers
* ChromaDB vector database integration
* Question answering using Groq LLM
* Website summarization
* Website statistics dashboard
* Chat history
* Source context display
* Quick Questions for faster interaction

## Technologies Used

* Python
* Streamlit
* BeautifulSoup
* Requests
* Sentence Transformers
* ChromaDB
* Groq API
* Git & GitHub

## Workflow

1. Enter a website URL
2. Scrape website content
3. Split content into chunks
4. Generate embeddings
5. Store embeddings in ChromaDB
6. Ask questions about the website
7. Retrieve relevant content
8. Generate answers using Groq LLM
9. Generate website summary

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ATQ-RAG-Website-Chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
ATQ-RAG-Website-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── screenshots/
```

## Screenshots

home.png
statistics.png
question_answer.png
summary.png
chat_history.png
quick_questions.png

## Future Enhancements

* Multi-website support
* PDF document support
* Multi-page website crawling
* Voice-based interaction
* Advanced analytics dashboard

## Author

Niranjan K D

B.Tech Computer Science Engineering

AI / ML Developer