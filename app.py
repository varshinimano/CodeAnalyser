
from langchain_chroma import Chroma
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv

import os
import shutil
import stat
import sys
import gc

from flask import Flask, render_template, jsonify, request
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


app = Flask(__name__)

load_dotenv()

#If the repo hash readonly permission means , we are changing it to write only permission and executing
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def create_rag_chain():

    embeddings = load_embedding()

    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8
        }
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given a chat history and the latest user question,
rewrite the question so that it can be understood independently.

Do not answer the question.
Only rewrite the question."""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an assistant that answers questions about a
GitHub code repository.

Use the retrieved code context to answer the question.

If you don't know the answer, say that you don't know.

Keep the answer clear and concise.

Retrieved code:

{context}"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    return create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )


rag_chain = None
chat_history = []
current_repo_url = None


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

#If the user send the url means , it comes like POST/chatbot(defined in html)
@app.route("/chatbot", methods=["GET", "POST"]) #when user gives url and clicks send
def gitRepo():

    global rag_chain
    global chat_history
    global current_repo_url

    if request.method != "POST":
        return jsonify({
            "response": "Please submit a GitHub repository URL."
        })

    user_input = request.form["question"].strip()

    if not user_input:
        return jsonify({
            "response": "Please submit a GitHub repository URL."
        }), 400

    repo_url = user_input.rstrip("/")

    # Same repository: reuse existing RAG chain
    if repo_url == current_repo_url and rag_chain is not None:

        print("Same repository. Reusing existing RAG chain.")

        chat_history = []

        return jsonify({
            "response": user_input 
            #flask sending back to frontend that i have processed the url 
        })

    print("New repository:", user_input)

    # Remove old RAG chain
    rag_chain = None
    chat_history = []
    gc.collect()

    # Remove old repository and database
    if os.path.exists("repo"):
        print("Removing old repository...")
        shutil.rmtree(
            "repo",
            onexc=remove_readonly
        )

    if os.path.exists("db"):
        print("Removing old Chroma database...")
        shutil.rmtree(
            "db",
            onexc=remove_readonly
        )

    # Clone new repository
    print("Cloning new repository...")
    repo_ingestion(user_input)

    # Create vector database
    print("Creating vector database...")

    result = os.system(
        f'"{sys.executable}" store_index.py' #see in the a same python env
    )

    print(
        "store_index.py exit code:",
        #exitcode = 0 means success
        #exitcode = 1 means failure
        result
    )

    if result != 0:
        return jsonify({
            "response": "Error while creating vector database."
        }), 500 #flask sending to frontend

    if not os.path.exists("db"):
        return jsonify({
            "response": "Vector database was not created."
        }), 500

    # Create RAG chain
    print("Loading new RAG chain...")

    rag_chain = create_rag_chain() 
    #just create like a template, when we invoke only it runs. 

    current_repo_url = repo_url
    chat_history = []

    print("New repository is ready!")

    return jsonify({
        "response": user_input
    })  #sending to frontend that successfully processed the url

#if the user send the msg means , it comes in POST/get(defined in html page)
@app.route("/get", methods=["GET", "POST"])
def chat():

    global chat_history

    if rag_chain is None: #if the user send msg without sending the url means
        return "Please submit a GitHub repository first."

    user_input = request.form["msg"]

    if user_input.lower() == "clear":

        chat_history = []

        return "Chat history cleared."

    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    answer = result["answer"]

    print("User:", user_input)
    print("Assistant:", answer)

    chat_history.append(
        ("human", user_input)
    )

    chat_history.append(
        ("ai", answer)
    )

    return str(answer)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )
