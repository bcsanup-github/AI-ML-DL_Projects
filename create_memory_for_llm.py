from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# step 1: load pdf

DATA_PATH=(r"C:\Users\user\Desktop\chatbotpdf\data\CSLP_Assignment.pdf")
def load_pdf_files(data,
                   glob='*.pdf',
                   loader_cls=PyPDFLoader):
    
    loader = loader_cls(data)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("length of pdf pages: ", len(documents))

# step 2: create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,
                                                 chunk_overlap=30)
    
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("length of text chunks: ", len(text_chunks))


# step3: create vector enbeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

#step 4: store embedding in FIASS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)