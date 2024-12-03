from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import random

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["GOOGLE_API_KEY"] = "AIzaSyBjkWt3xOEnRaYVB-LUYSrUeE1rEFgjA6U"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "mbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Multiple prompt templates for variability
GENERALIZATION_PROMPTS = [
    "Provide a comprehensive and generalized explanation based on the context.",
    "Synthesize the information into a broad, insightful overview.",
    "Extract key insights and present a generalized perspective.",
    "Construct a nuanced, well-rounded explanation transcending specific details."
]

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7, max_tokens=5000)

def get_dynamic_prompt():
    generalization_instruction = random.choice(GENERALIZATION_PROMPTS)
    return ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n{generalization_instruction}"),
        ("human", "{input}"),
    ])

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        
        # Dynamic prompt generation
        dynamic_prompt = get_dynamic_prompt()
        question_answer_chain = create_stuff_documents_chain(llm, dynamic_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)