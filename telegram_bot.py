import os
from dotenv import load_dotenv
import shutil
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app import extract_text_from_pdf,get_text_chunks,get_vector_store,get_conversational_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
import string

load_dotenv()

def generate_random_filename(length=10):
    characters = string.ascii_letters + string.digits  # Letters and digits
    filename = ''.join(random.choice(characters) for _ in range(length))
    return filename

# Set up Telegram bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Set up Google Generative AI API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize FAISS storage
VECTOR_STORE_PATH = generate_random_filename(12)
vector_store = None

# Function to process and store embeddings
async def process_pdf_and_store_vectors(pdf_path):
    global vector_store

    text = extract_text_from_pdf(pdf_path)
    text_chunks = get_text_chunks(text)
    vector_store = get_vector_store(text_chunks,VECTOR_STORE_PATH)
   
# Function to load FAISS index
def load_vector_store():
    global vector_store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        vector_store = None

# Function to delete FAISS index
def delete_faiss_index():
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)

# Telegram command: Start
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! Send me a PDF to upload, and then you can chat with it.")

# Telegram command: Upload PDF
async def handle_pdf(update: Update, context: CallbackContext):
    file = await update.message.document.get_file()
    file_path = f"./uploads/{file.file_id}.pdf"
    
    os.makedirs("uploads", exist_ok=True)
    await file.download_to_drive(file_path)
    
    await update.message.reply_text("Processing PDF... This may take a few seconds.")
    await process_pdf_and_store_vectors(file_path)
    
    await update.message.reply_text("PDF uploaded and indexed successfully! Now, you can chat with it.")

# Telegram command: Chat with PDF
async def chat_with_pdf(update: Update, context: CallbackContext):
    query = update.message.text
    if vector_store is None:
        await update.message.reply_text("No PDF uploaded yet! Please send a PDF first.")
        return

    results = vector_store.similarity_search(query)
    chain = get_conversational_chain()
    
    response =chain.invoke({"input_documents":results,"question":query})
    response_text = response.get("output_text", "No relevant information found.")
    
    # Send the response as text
    await update.message.reply_text(response_text)
    # await update.message.reply_text(response)

# Telegram command: Reset (delete stored PDFs)
def reset(update: Update, context: CallbackContext):
    delete_faiss_index()
    update.message.reply_text("All uploaded PDFs and stored data have been deleted.")

# Main function to start the bot
def main():
    app = Application.builder().token("7400660082:AAFZTLVwV6yYqUV99iZnc-1PLNUlWy1SkG8").build()
    # dp = updater.dispatcher

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_pdf))


    load_vector_store()
    app.run_polling()
    # updater.idle()

if __name__ == "__main__":
    main()
