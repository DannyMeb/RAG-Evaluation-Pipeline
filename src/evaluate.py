# Project: Rag systems evaluation pipeline using Ragas
# Authored by Daniel
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.

import os
import sys
import json
import pandas as pd
import requests
from plotter import plot_metrics, compare_models_metrics, plot_category_distribution
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio

# Apply nest_asyncio
nest_asyncio.apply()

# Set Constants when running this script in isolate mode
JSON_FILE_PATH = '/home/ubuntu/ragas/synthetic_testset/Synthetic_testset.json'
BASE_URL_DIFY = "http://98.83.213.54/v1/"
BASE_URL_RAGFLOW = "http://98.83.213.54/v1/api/"

RAGFLOW_API_KEY = 'ragflow-c2OGIwMmVjN2E2NzExZWY4M2U3MDI0Mm'  # Replace with your actual RAGFlow API Key
DIFY_API_KEY="app-JUx44q0XLkn1jyZx6uU73Yrv"
GENERATED_ANSWERS_JSON_PATH = '/home/ubuntu/ragas/synthetic_testset/Synthetic_testset_with_answers.json'
RESULTS_EXCEL_PATH = '/home/ubuntu/ragas/synthetic_testset/benchmarks.xlsx'
BENCHMARKS_JSON_PATH = '/home/ubuntu/ragas/synthetic_testset/benchmarks.json'
METRICS_PLOT_PATH = '/home/ubuntu/ragas/synthetic_testset/average_metrics_plot.png'
OPENAI_API_KEY = "sk-None-Rh3Lxa0sHMVKO4JvOcejT3BlbkFJEL64fWxoupPyPinn2TDS"  

#Set payload HEADERS

def get_base_url(base_url, system_name):
    if system_name == 'ragflow':
        return f"{base_url}/api"
    elif system_name == 'dify':
        return f"{base_url}"




def set_headers(rag_api_key):
    headers = {
        'Authorization': f'Bearer {rag_api_key}',
        'Content-Type': 'application/json'
    }

    return headers



def set_openai_api_key(openAi_api_key):
    os.environ["OPENAI_API_KEY"] = openAi_api_key


def create_conversation(base_url, api_key, system_name, user_id):
    if system_name == 'ragflow':
        # RagFlow uses GET to create a conversation
        # user_id=""
        url = f"{base_url}new_conversation"
        headers = set_headers(api_key)
        params = {"user_id": user_id}

        response = requests.get(url, headers=headers, params=params)  # GET for RagFlow
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json['retcode'] == 0:
                return response_json['data']['id']
        print(f"Error creating conversation for user {user_id}: {response.status_code}")
        return None

    elif system_name == 'dify':
        # DigFy does not require a separate API call to create a conversation
        # Return an empty conversation_id for DigFy since it will handle it dynamically
        return ""




def get_generated_answer(base_url, api_key, conversation_id, question, system_name, user_id):
    headers = set_headers(api_key)

    if system_name == 'ragflow':
        # RagFlow uses POST to generate answers
        url = f"{base_url}/completion"
        # print(url)
        payload = {
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": question}],
            "quote": False,
            "stream": False
        }

    elif system_name == 'dify':
        # DigFy uses POST for both creating a conversation and generating an answer
        url = f"{base_url}chat-messages"  # For DigFy, the base URL directly points to the chat endpoint
        payload = {
            "inputs": {},
            "query": question,
            "stream": False, 
            "conversation_id": conversation_id,  # Empty for new conversation
            "user": user_id
        }

    # Both RagFlow and DigFy use POST for generating answers
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        if system_name == 'ragflow':
            return response_data['data']['answer']
        elif system_name == 'dify':
            return response_data['answer']
    else:
        print(f"Error generating answer for question '{question}' using {system_name}. Status code: {response.status_code}")
        return None



def load_and_process_dataset(base_url,rag_system_name, rag_api_key,json_file_path):
    # Load dataset from JSON
    print(f"Loading dataset from {json_file_path}...")
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Process questions and generate answers
    print("Generating answers for the entire dataset...")
    df_with_answers = process_questions(base_url,rag_system_name,rag_api_key, df)

    return df_with_answers


def process_questions(base_url, rag_system_name, rag_api_key, df):
    generated_answers = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Answers", unit="question"):
        question = row['question']
        user_id = f'test_user_{index}'
        conversation_id = create_conversation(base_url, rag_api_key,rag_system_name, user_id)
        
        if conversation_id is not None:
            generated_answer = get_generated_answer(base_url, rag_api_key, conversation_id, question, rag_system_name, user_id)
            generated_answers.append(generated_answer)
        else:
            generated_answers.append(None)

    df.loc[:, 'answer'] = generated_answers  # Avoid SettingWithCopyWarning
    return df



def save_generated_answers(df, output_file):
    data_with_answers = df.to_dict(orient='records')
    with open(output_file, 'w') as json_file:
        json.dump(data_with_answers, json_file, indent=4)
    print(f"Dataset with generated answers saved to {output_file}")


def load_dataset_for_evaluation(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data_with_answers = json.load(json_file)
    df_with_answers = pd.DataFrame(data_with_answers)
    return df_with_answers


def prepare_dataset_for_evaluation(df):
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    contexts = df['contexts'].apply(lambda x: [x] if isinstance(x, str) else x).tolist()
    ground_truths = df['ground_truth'].tolist()

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    return Dataset.from_dict(data), df


def evaluate_dataset(dataset_tuple):
    dataset, original_df = dataset_tuple
    print("Evaluating dataset with Ragas metrics...")
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision]
    )
    result_df = result.to_pandas()
    
    # Merge evaluation results with original dataframe
    merged_df = pd.concat([result_df, original_df[['evolution_type', 'metadata', 'episode_done']]], axis=1)
    
    return merged_df


def save_evaluation_results(result_df, excel_path, json_path):
    result_df.to_excel(excel_path, index=False)
    result_df.to_json(json_path, orient='records', indent=4)
    print(f"Results saved as Excel ({excel_path}) and JSON ({json_path}).")




def main():
    # Present the user with a numbered menu
    print("Choose the system:")
    print("1. RagFlow")
    print("2. Dify")
    
    # Get user input as a number
    choice = input("Enter your choice (1 or 2): ").strip()

    # Determine system name based on the user's choice
    if choice == '1':
        system_name = 'ragflow'
        print("You have selected RagFlow.")
    elif choice == '2':
        system_name = 'dify'
        print("You have selected Dify.")
    else:
        print("Invalid choice. Please enter 1 for RagFlow or 2 for Dify.")
        return

    # Get the correct base URL based on the system and IP address
    base_url = ""

    # Set the correct API key based on the system
    if system_name == 'ragflow':
        rag_api_key = RAGFLOW_API_KEY  # Ensure RAGFLOW_API_KEY is defined elsewhere
        base_url = BASE_URL_RAGFLOW
    elif system_name == 'dify':
        rag_api_key = DIFY_API_KEY  # Ensure DIGFY_API_KEY is defined elsewhere
        base_url=BASE_URL_DIFY
    # Optional: Set OpenAI API key if needed (depending on your system logic)
    set_openai_api_key(OPENAI_API_KEY)

    # Load and process the dataset (using the chosen system and API key)
    df_with_answers = load_and_process_dataset(base_url,system_name, rag_api_key, JSON_FILE_PATH)

    # Save the dataset with generated answers
    save_generated_answers(df_with_answers, GENERATED_ANSWERS_JSON_PATH)

    # Load dataset for evaluation
    df_with_answers = load_dataset_for_evaluation(GENERATED_ANSWERS_JSON_PATH)

    # Prepare the dataset for evaluation
    dataset = prepare_dataset_for_evaluation(df_with_answers)

    # Perform the evaluation
    result = evaluate_dataset(dataset)

    # Save evaluation results in Excel and JSON formats
    save_evaluation_results(result, RESULTS_EXCEL_PATH, BENCHMARKS_JSON_PATH)

    # Plot and save the evaluation metrics
    plot_metrics(result, METRICS_PLOT_PATH)

    print("Process completed successfully!")


if __name__ == "__main__":
    main()


