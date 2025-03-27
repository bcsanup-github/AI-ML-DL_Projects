import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

#step1: setup llm(mistral with huggingface)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7b-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"200"}
    )
    return llm

#step2: connect llm with faiss and create chain


CUSTOM_PROMPT_TEMPLATE = """
use the piece of information provided in the context to answer user's question.
if you don't know the answer just say i don't know,don't try to make up an answer.
don't say anything out of the context.

Context:{context}
Question:{question}

start the answer , no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

#load database
DB_FAISS_PATH=r"vectorstore/db_faiss"
embedding_model= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)

#creating QA chain
llm = load_llm(HUGGINGFACE_REPO_ID)
qa_chain= RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k':3}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}

)

#invoke
user_query=input("Write Query Here: ")
response=qa_chain.invoke({"query": user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS: ", response.get("source_documents", "No source documents available"))