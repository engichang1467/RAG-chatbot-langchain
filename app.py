from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, pipeline
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_TOKEN']

# Load the data
loader = HuggingFaceDatasetLoader(path="databricks/databricks-dolly-15k", page_content_column="context", use_auth_token=hf_api_key)
data = loader.load()

# Document Transformers
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# Text Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Set up Vector Stores
db = FAISS.from_documents(docs, embeddings)

# Set up retrievers
retriever = db.as_retriever(search_kwargs={"k": 4})

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert", padding=True, truncation=True, max_length=512)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering",
    model="Intel/dynamic_tinybert",
    tokenizer=tokenizer,
    return_tensors='pt'
)

def generate(question):
    docs = retriever.get_relevant_documents(question)
    context = docs[0].page_content
    squad_ex = question_answerer(question=question, context=context)
    return squad_ex['answer']


def respond(message, chat_history):
    bot_message = generate(message)
    chat_history.append((message, bot_message))
    
    return "", chat_history

# Set up the chat interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Ask away")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

demo.queue().launch()