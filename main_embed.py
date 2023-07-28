import arxiv
import requests
import PyPDF2
from io import BytesIO
import tiktoken
import os
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
openai.api_key = os.getenv("OPENAI_API_KEY")


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def sanitize_filename(filename):
    invalid_chars = set(r'\/:*?"<>|')
    sanitized_filename = "".join(c for c in filename if c not in invalid_chars)
    sanitized_filename = "_".join(sanitized_filename.split())
    return sanitized_filename

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_article_pdf(url):
    response = requests.get(url)
    pdf = PyPDF2.PdfReader(BytesIO(response.content) )
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def sanitize_article_text(text):
    references_index = text.upper().find("REFERENCES")
    if references_index != -1:
        text = text[:references_index]
    return text

def save_article(save_path, text):
    with open(save_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)

def summarize_article(text):
    # Check the number of tokens in the text
    num_tokens = count_tokens(text)

    # Limit the text to the first 15,000 tokens if it exceeds that limit
    if num_tokens > 15000:
        firsttext = text[:7500] # Get first 7500 tokens
        lasttext = text[7500:] # Get last 7500 tokens
        text = firsttext + lasttext # Concatenate first 7500 tokens with first 7500 tokens of last 7500 tokens
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        stream=True,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a helpful assistant which summarizes scientific papers in an easy and understandable concise way. your goal is to create a summary, create up to 10 bullet points about the most important things about the paper."},
            {"role": "user", "content": f"Please return a summary the following text and extract up to 10 bullet points and 10 relevant keywords: {text}"}
        ]
    )

    responses = ''
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(r_text, end='', flush=True)
    return responses

def get_embedding(text):
    response = openai.Embedding.create(
        input=text, 
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def search_similar_articles(query, df):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], df["embedding"].tolist())
    top_index = similarities[0].argmax()
    return df.iloc[top_index]

def search_semantic_scholar(query,year,n=100):
    current_index = 0
    r = requests.get(
    'https://api.semanticscholar.org/graph/v1/paper/search?query='+query+'&year='+year+'-',
    params={'fields': 'title,year,abstract,url,openAccessPdf,tldr,isOpenAccess,publicationDate'}
    )
    #print(json.dumps(r.json(), indent=2))
    r = r.json()
    total = r['total']
    offset = r['offset']
    #nextindex = r['next']
    final_results = pd.json_normalize(r['data'])
    current_index = offset+n+1
    print('Total papers: ',total)
    print('Current index: ',current_index)
    while current_index < total:
        r = requests.get(
        'https://api.semanticscholar.org/graph/v1/paper/search?query='+query+'&offset='+str(current_index)+'&year='+year+'-'+'&limit='+str(n),
        params={'fields': 'title,year,abstract,url,openAccessPdf,tldr,isOpenAccess,publicationDate'}
        )
        #print(json.dumps(r.json(), indent=2))
        r = r.json()
        offset = r['offset']
        #nextindex = r['next']
        results = pd.json_normalize(r['data'])
        final_results = pd.concat([final_results, results], ignore_index=True)
        current_index = offset+n+1
        print('Current index: ',current_index)
    return final_results



def main(keyword, n, save_directory,year):
    create_directory(save_directory)
    saved_filenames = set(os.listdir(save_directory))
  
    results = search_semantic_scholar(keyword,year,n)
    print(results)
    print(results['url'].to_string()  ) 
    df_old = pd.DataFrame()
    # if csv file exists, read it in
    if os.path.exists("summary_embeddings.csv"):
        df_old = pd.read_csv("summary_embeddings.csv")

    df_new = pd.DataFrame(columns=["title", "summary", "url", "embedding","publicationDate","AISummaryAvailable"])
    i=0
    
    for ind in results.index:
        title = results['title'][ind]
        abstract = results['abstract'][ind]
        openAccessPdf = results['openAccessPdf.url'][ind]
        url = results['url'][ind]
        tldr = results['tldr.text'][ind]
        isOpenAccess = results['isOpenAccess'][ind]
        publicationDate = results['publicationDate'][ind]
        print("Title: "+title)
        print("URL: "+url)
        


        filename = sanitize_filename(title) + ".txt"
        if filename in saved_filenames:
            print(f"Article {i+1} already saved.")
            continue
        
        try:
            #If article is open access, with a condition to catch non-readable PDFs (in which case use abstract instead)
            if isOpenAccess == False or openAccessPdf == None or openAccessPdf=='':
                text = tldr
            else:
                text = download_article_pdf(openAccessPdf)
            
            # print the token count of the article
            print(f"Article {i+1} has {count_tokens(text)} tokens.")
            text = sanitize_article_text(text)
            # print the token count of the article after sanitization
            print(f"Article {i+1} has {count_tokens(text)} tokens after sanitization.")
            save_path = os.path.join(save_directory, filename)
            save_article(save_path, text)
            summary = summarize_article(text)
            embedding = get_embedding(summary)
            # append each new article to the df_new dataframe
            if openAccessPdf==None or openAccessPdf=='' or isOpenAccess == False:
                df_new = df_new.append({"title": title, "summary": summary, "url": url, "embedding": embedding,"publicationDate": publicationDate, "AISummaryAvailable": 0}, ignore_index=True)
            else: 
                df_new = df_new.append({"title": title, "summary": summary, "url": openAccessPdf, "embedding": embedding,"publicationDate": publicationDate, "AISummaryAvailable": 1}, ignore_index=True)
            print(f"\nSummary of article {i+1}:\n{summary}\n")
            summary_filename = filename.replace(".txt", "_summary.txt")
            summary_save_path = os.path.join(save_directory, summary_filename)
            save_article(summary_save_path, summary)
            i=i+1
        except Exception as e:
            #If article is NOT open access, use abstract instead
            print('Could not load article, defaulting to using abstract')
            if tldr != None:
                text = str(tldr)
            elif abstract != None:
                text=str(abstract)
            else:
                text=str(title)

            save_path = os.path.join(save_directory, filename)
            save_article(save_path, text)
            summary = text
            embedding = get_embedding(summary)
            # append each new article to the df_new dataframe
            df_new = df_new.append({"title": title, "summary": summary, "url": url, "embedding": embedding,"publicationDate": publicationDate,"AISummaryAvailable":0}, ignore_index=True)
            print(f"\nSummary of article {i+1}:\n{summary}\n")
            summary_filename = filename.replace(".txt", "_summary.txt")
            summary_save_path = os.path.join(save_directory, summary_filename)
            save_article(summary_save_path, summary)
            i=i+1
            print('End')

    # concatenate new dataframe (df_new) with old dataframe (df_old), with new data on top
    df_combined = pd.concat([df_new, df_old], ignore_index=True)
    df_combined['date'] = pd.to_datetime(df_combined['publicationDate'])
    df_combined= df_combined.sort_values(by='date', ascending=False)

    df_combined.to_csv("summary_embeddings.csv", index=False)



if __name__ == "__main__":
    keyword = "HER3 ErBb3"
    year = str(2023)
    n = 100
    save_directory = "saved_articles"
    main(keyword, n, save_directory,year)
