# Project: Rag systems evaluation pipeline using Ragas
# Authored by Daniel
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.


import os
import ragas
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.testset.evolutions import simple, reasoning, multi_context
from pydantic import BaseModel
import pandas as pd
import json
import nest_asyncio

# Apply nest_asyncio
nest_asyncio.apply()

# OpenAI API Key setup
def setup_api_keys(openAi_api_key):
    os.environ['USER_AGENT'] = 'myagent'
    os.environ["OPENAI_API_KEY"] = openAi_api_key
    print("API Key exported ...")

# Configure generator models
def configure_generator_models():
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
    critic_llm = ChatOpenAI(model="gpt-4")
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': "cuda"}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("Generator models configured successfully...")
    return generator_llm, critic_llm, embeddings

# Load documents from a directory
def load_documents(directory="data"):
    loader = DirectoryLoader(directory, recursive=True)
    documents = loader.load()    
    for document in documents:
        document.metadata['filename'] = document.metadata['source']
    print(f"{len(documents)} documents loaded successfully.")
    return documents

# Initialize the test set generator
def initialize_generator(generator_llm, critic_llm, embeddings):
    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    print("Generator initialization completed...")
    return generator

# Generate the test set
def generate_testset(generator, documents, num_samples=20, proportions=None):
    if proportions is None:
        proportions = {simple: 0.34, reasoning: 0.33, multi_context: 0.33}
    
    testset = generator.generate_with_langchain_docs(documents, test_size=num_samples, distributions=proportions)
    df = testset.to_pandas()
    print("Test set generated successfully...")
    return df

# Save the test set in different formats
def save_testset(df, json_filename="synthetic_testset/Synthetic_testset.json"):
    df.to_json(json_filename, orient="records", indent=2)
    print(f"JSON file saved as {json_filename}")
    print("Test set saved successfully.")

# Main function to run the pipeline
def main(num_samples=20, proportions=None):
    setup_api_keys()
    generator_llm, critic_llm, embeddings = configure_generator_models()
    documents = load_documents()
    generator = initialize_generator(generator_llm, critic_llm, embeddings)
    df = generate_testset(generator, documents, num_samples, proportions)
    save_testset(df)

if __name__ == "__main__":
    # Example of running the script with default values
    main(num_samples=3, proportions={simple: 0.4, reasoning: 0.3, multi_context: 0.3})
