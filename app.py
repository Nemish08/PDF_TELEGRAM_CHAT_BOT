from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAoGDaapebswJiT8RQ24NRZvNW6Gc3zJ14"

uploaded_file = "test.pdf"
user_question = "how are you?"

def extract_text_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print("text extract from pdf (done)")
    return text

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    print("get chunks (done)")
    return text_splitter.split_text(text)

def get_vector_store(text_chunks,vector_store_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_path)
    print("get vector store : (done)")
    return vector_store

def get_conversational_chain():
    """
    Creates a conversational chain for QA using LangChain and Google Generative AI.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If qestion ask about explain pdf or sumthing related to pdf  then give answer about context summerization. otherwise you will be panalized
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


if __name__=="__main__":

    if uploaded_file and user_question:
        try:
            # Step 1: Extract text from PDF
            raw_text = extract_text_from_pdf(uploaded_file)

            # Step 2: Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Step 3: Embed text into a vector store
            vector_store = get_vector_store(text_chunks)

            # Step 4: Perform similarity search and generate the answer
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            print(response['output_text'])
        except Exception as e:
            print("error : ",e)
