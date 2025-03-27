import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"200"}
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'message' not in st.session_state.messages:
        st.session_state.message = []

    for message in st.session_state.message:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.message.append({'role':'user', 'content':'prompt'})

        CUSTOM_PROMPT_TEMPLATE = """
            use the piece of information provided in the context to answer user's question.
            if you don't know the answer just say i don't know,don't try to make up an answer.
            don't say anything out of the context.

            Context:{context}
            Question:{question}

            start the answer , no small talk.
            """
        HF_TOKEN=os.environ.get("HF_TOKEN")
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7b-Instruct-v0.3"
        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

        try:
            vectorestore=get_vectorstore()
            if vectorestore is None:
                st.error("failed to load vectorstore ")

           
            qa_chain= RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
                llm=llm,
                retriever=db.as_retriever(search_kwargs={'k':3}),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
                
            

            response=qa_chain.invoke({"query": prompt})

            result:response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+str(source_documents)
            st.chat_message("assistant").markdown(response)
            st.session_state.message.append({'role':'assistance', 'content':'response'})


        except Exception as e:
            st.error(f"Error:{str(e)}")

if __name__ == "__main__":
    main()