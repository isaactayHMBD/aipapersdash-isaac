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
    'fieldnames': ['abstract', 'keypoints', 'target']
})
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)

def retrieve_info(query):
    response = db.similarity_search(query, 
                                    k=3) #retrieves top 3
    contents = [doc.page_content for doc in response]
    return contents

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are an expert summarizer on breast cancer research papers. I am going to share with you an abstract of an academic paper
and you will give me a summarization of the most relevant information in 10 bullet points or less, based on past
summaries. The past summaries are 
You will follow all the rules below: 

1/ You must identify and know what the protein receptors (targets) do.
2/ Response must be similar to past summaries, in terms of length, and tone of voice. Summary should also mention what the targets are doing.
3/ You must list all the Programs (ADC) and targets (HER3/ErbB3, etc.) that the paper has mentioned, after the bullet points are formed
4/ If the summaries are found to be irrelevant to the abstract, or if new information is found in the abstract, please mention the new information in the bullet points


Below is the abstract of an academic paper on cancer: 
{message}

Here is a list of how the summaries would be like for a similar paper: 
{summary}

Please write the summary for the abstract.
"""

prompt = PromptTemplate(
    input_variables=["message","summary"],
    template=template

)
chain = LLMChain(llm=llm, prompt=prompt)

def load_pdf():
    text=""
    loader = PyPDFDirectoryLoader(".", glob="**/[!.]*.pdf")
    for page in loader.load():
       text+=page.page_content
    return text

def generate_response(message):
    summary=retrieve_info(message)
    response = chain.run(message=message, summary=summary)
    print(response)
    return response

message = """
Article
Head and neck squamous cell carcinoma (HNSCC) is the sixth most common cancer type, has often an aggressive course and is poorly responsive to current therapeutic approaches, so that 5-year survival rates for patients diagnosed with advanced disease is lower than 50%. The Epidermal Growth Factor Receptor (EGFR) has emerged as an established oncogene in HNSCC. Indeed, although HNSCCs are a heterogeneous group of cancers which differ for histological, molecular and clinical features, EGFR is overexpressed or mutated in a percentage of cases up to about 90%. Moreover, aberrant expression of the other members of the ErbB receptor family, ErbB2, ErbB3 and ErbB4, has also been reported in variable proportions of HNSCCs. Therefore, an increased expression/activity of one or multiple ErbB receptors is found in the vast majority of patients with HNSCC. While aberrant ErbB signaling has long been known to play a critical role in tumor growth, angiogenesis, invasion, metastatization and resistance to therapy, more recent evidence has revealed its impact on other features of cancer cells' biology, such as the ability to evade antitumor immunity. In this paper we will review recent findings on how ErbB receptors expression and activity, including that associated with non-canonical signaling mechanisms, impacts on prognosis and therapy of HNSCC.

“ErbB3 appears to play different roles in HNSCC progression depending on its membranous/cytoplasmic vs. nuclear localization. Unlike nuclear EGFR, localization of ErbB3 and ErbB4 in the nucleus of HNSCC cells appears associated with a more favorable prognosis. EGFR-positive laryngeal tumors co-expressing nuclear ErbB3 had a better prognosis as compared to those that expressed EGFR without ErbB3 or in association with cytoplasmic ErbB3. Moreover, based on the observation that ErbB3 was never expressed alone, but always co-expressed with ErbB2, both in the presence and absence of EGFR, the authors of these studies speculated that ErbB3 nuclear localization may play a favorable role by preventing the formation of ErbB heterodimers at the cell membrane and the ensuing activation of pro-tumoral downstream signaling pathways.”

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

