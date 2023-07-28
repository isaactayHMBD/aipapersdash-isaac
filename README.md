# aipapersdashboard
Publication dashboard powered by LLM model to summarize papers for a given keyword search

## Overview
- Searches publications from semantic scholar's API using a keyword search on a daily basis
- These are then parsed to identify open access vs non-open access publications, publication date, titles, abstracts and other key information. 
- Publications which have open access have PDF's retrieved and parsed, the text is fed into GPT3.5 for summarization
- Publications which do not have open access have abstract summaries generated instead


To install the dependencies run pip install -r requirements.txt 
To run use uvicorn webapp:app --host 127.0.0.1 --port 8000 to run locally or modify port and host to suit your needs

webapp.py runs the website
main_embed.py is called from webapp.py to perform pull down of papers and embedding into GPT3.5 tokens

Paper summaries are saved in saved_articles and a running summary is saved in summary_embeddings

Keyword searches should be set in webapp.py



### Future work
- Extend to multiple dashboards for different pre-defined keyword searches
- Email updates to relevant individuals as publications are generated on a regular basis
- Add functionality to manually upload a set of papers which can be queried