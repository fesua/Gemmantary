from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import torch
import pandas as pd

import json
from pathlib import Path
from pprint import pprint

class AIAgent:
    """
    Gemma 2b-it assistant.
    It uses Gemma transformers 2b-it/3.
    """
    def __init__(self, model_path, max_length=1000):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gemma_lm = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")

    def create_prompt(self, query, video_caption, context):
        # prompt template
        prompt = f"""
        You are an AI agent specialized in creating commentary script of live sports events.
        Describe the game using the video caption provided (Video Caption).
        Describe the sport and the player using the context provided (Context).
        In order to create the commentary, please use the information from the context provided (Context).
        If needed, include also explanations.
        Video Caption: {video_caption}
        Question: {query}
        Context: {context}
        Answer:
        """
        return prompt
    
    def generate(self, query, video_caption, retrieved_info):
        prompt = self.create_prompt(query, video_caption, retrieved_info)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        # Answer generation
        answer = self.gemma_lm.generate(
            input_ids,
            #max_length=self.max_length, # limit the answer to max_length
            max_new_tokens=self.max_length
        )
        # Decode and return the answer
        answer = self.tokenizer.decode(answer[0], skip_special_tokens=True, skip_prompt=True)
        return prompt, answer
    
class RAGSystem:
    """Sentence embedding based Retrieval Based Augmented generation.
        Given database of pdf files, retriever finds num_retrieved_docs relevant documents"""
    def __init__(self, ai_agent, rag_path, num_retrieved_docs=1):
        # load the data
        self.num_docs = num_retrieved_docs
        self.ai_agent = ai_agent
        if '.csv' in rag_path:
            loader = CSVLoader(rag_path)
        else:
            loader = JSONLoader(file_path=rag_path, jq_schema='.documents[].content')
            
        documents = loader.load()
        self.template = "\n\nQuestion:\n{question}\n\nPrompt:\n{prompt}\n\nAnswer:\n{answer}\n\nContext:\n{context}"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        # create a vectorstore database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                           model_kwargs = {"device": "cuda"})
        self.vector_db = Chroma.from_documents(documents=all_splits, 
                                               embedding=embeddings, 
                                               persist_directory="chroma_db")
        self.retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={'k': self.num_docs})

    def retrieve(self, query):
        # retrieve top k similar documents to query
        docs = self.retriever.get_relevant_documents(query)
        return docs
    
    def query(self, query, video_caption):
        # generate the answer
        context = self.retrieve(query)
        data = ""
        for item in list(context):
            data += item.page_content
            
        data = data[:5000]

        prompt, answer = self.ai_agent.generate(query, video_caption, data)
        
        return self.template.format(question=query,
                                    prompt=prompt,
                                   answer=answer,
                                   context=context)