import sys
import arxiv
import requests
import PyPDF2
from io import BytesIO
import tiktoken
import os
from dotenv import load_dotenv
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain.text_splitter import CharacterTextSplitter
import sentence_transformers
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, CSVLoader
from langchain.indexes import VectorstoreIndexCreator
import getpass

load_dotenv()

loader = CSVLoader(file_path='key_points.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['target/program', 'title', 'abstract', 'keypoints']
})
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)

def retrieve_info(query):
    response = db.similarity_search(query, 
                                    k=2) #retrieves top 2
    contents = [doc.page_content for doc in response]
    return contents

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are an expert summarizer on breast cancer and genomic research papers. I will share with you an abstract of an academic paper
and you will give me key points that I am interested in, which are based on past examples. Key points will be in 10 or less bullet points.

You will follow all the rules below: 
- Identify and know what are the cancer therapeutic targets acronyms, and what each one does. This will be your context. You do not need to mention it in the bullet points.
- Response must be similar to past summaries, mentioning what the targets are doing, pros and cons, new findings/discoveries,
how the therapeutic targets interact with other targets.
- If new information is found in the abstract when compared to the summary, please mention the new information from the abstract in the bullet points.
- Bullet points should only include information from the abstract.
- Do not paraphrase too much. Bullet points must not repeat.
- You must list all the therapeutic targets (HER3/ErbB3, etc.) that the paper has mentioned, after the bullet points are formed.

Below is the abstract of an academic paper on cancer: 
{message}

Here are some key points that I will be interested in, that might be in the paper: 
{summary}

Please pick out the key points from the abstract.
"""

prompt = PromptTemplate(
    input_variables=["message","summary"],
    template=template

)
chain = LLMChain(llm=llm, prompt=prompt)

#not in use yet
def load_pdf():
    text=""
    loader = PyPDFDirectoryLoader(".", glob="**/[!.]*.pdf")
    for page in loader.load():
       text+=page.page_content
    return text

#generate a response based on similarit
def generate_response(message):
    summary=retrieve_info(message)
    response = chain.run(message=message, summary=summary)
    print(response)
    return response

message = """
Article
Limited treatment options exist for EGFR-mutated NSCLC that has progressed after EGFR TKI and platinum-based chemotherapy. HER3 is highly expressed in EGFR-mutated NSCLC, and its expression is associated with poor prognosis in some patients. Patritumab deruxtecan (HER3-DXd) is an investigational, potential first-in-class, HER3-directed antibody–drug conjugate consisting of a HER3 antibody attached to a topoisomerase I inhibitor payload via a tetrapeptide-based cleavable linker. In an ongoing phase I study, HER3-DXd demonstrated promising antitumor activity and a tolerable safety profile in patients with EGFR-mutated NSCLC, with or without identified EGFR TKI resistance mechanisms, providing proof of concept of HER3-DXd. HERTHENA-Lung01 is a global, registrational, phase II trial further evaluating HER3-DXd in previously treated advanced EGFR-mutated NSCLC.
“In the ongoing phase I U31402-A-U102 trial, HER3-DXd 5.6 mg/kg demonstrated promising antitumor efficacy across a broad range of HER3 expression and a tolerable safety profile in heavily pretreated patients with EGFR-mutated NSCLC (N = 57).”

"""
generate_response(message) 

def get_text_chunks(raw_texts):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_texts)
    return chunks
    # chunk size -> how many characters in each chunk
    # chunk overlap -> how many characters the next chunk will be negatively offset
    # separator -> 

def get_vectorstore(text_chunks):
    embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embedding_model) #not persistent
    return vectorstore

def sanitize_article_text(text):
    references_index = text.upper().find("REFERENCES")
    if references_index != -1:
        text = text[:references_index]
    return text

# from langchain.document_loaders import OnlinePDFLoader 
# loader = PyPDFDirectoryLoader(".", glob="**/[!.]*.pdf")
# query = sys.argv[1]
# index = VectorstoreIndexCreator().from_loaders([loader])
# print(index.query(query))

