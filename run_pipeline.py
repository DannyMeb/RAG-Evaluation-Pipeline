# Project: Rag systems evaluation pipeline using Ragas
# Authored by Daniel
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.


import os
import time
import sys
from datetime import datetime
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.generator import setup_api_keys, configure_generator_models, load_documents, initialize_generator, generate_testset, save_testset
from src.evaluate import set_openai_api_key, load_and_process_dataset, save_generated_answers, load_dataset_for_evaluation, prepare_dataset_for_evaluation, evaluate_dataset, save_evaluation_results
from src.plotter import plot_metrics, compare_models_metrics, plot_category_distribution
from ragas.testset.evolutions import simple, reasoning, multi_context
import pandas as pd

# Centralized constants
BASE_DIR = '/home/ubuntu/ragas'
SYNTHETIC_TESTSET_DIR = f'{BASE_DIR}/synthetic_testset'
RESULTS_DIR = f'{BASE_DIR}/results'
JSON_FILE_PATH = f'{SYNTHETIC_TESTSET_DIR}/Synthetic_testset.json'
GENERATED_ANSWERS_JSON_TEMPLATE = f'{RESULTS_DIR}/{{system_name}}/{{model_name}}/Synthetic_testset_with_answers_{{model_name}}.json'
RESULTS_EXCEL_TEMPLATE = f'{RESULTS_DIR}/{{system_name}}/{{model_name}}/benchmarks_{{model_name}}.xlsx'
BENCHMARKS_JSON_TEMPLATE = f'{RESULTS_DIR}/{{system_name}}/{{model_name}}/benchmarks_{{model_name}}.json'
METRICS_PLOT_TEMPLATE = f'{RESULTS_DIR}/{{system_name}}/{{model_name}}/average_metrics_plot_{{model_name}}.png'
LOGS_DIR = '/home/ubuntu/ragas/logs'


# API Keys and Directories
BASE_URL_DIFY = "http://54.89.93.41/v1/"
BASE_URL_RAGFLOW = "http://54.89.93.41/v1/api/"

OPENAI_API_KEY = "Your-key"  
SOURCE_DATA_PATH = 'data'  

ragflow_models_api_keys = {
    "Falcon1-7B": "ragflow-YzYzBjZjA4ODE1NjExZWZiNDE1MDI0Mm",
    "Falcon2-11B": "ragflow-Q2ODg2NDc4N2IxMDExZWY4ODEwMDI0Mm",
    "Llama3.1": "ragflow-MyMzUyYTEwN2IxMDExZWY5YzNlMDI0Mm",
    "Llama3.2": "ragflow-BkMzA2ZmU4ODIyZTExZWZiYTBmMDI0Mm"
    }

dify_models_api_keys = {
    "Falcon1-7B":"app-GBNGmRBIMjU4dMPJNBW0WTQ4",
    "Falcon2-11B":"app-4S96CfMx8wCzb7f1EzR3cEoK",
    "Llama3.1": "app-wXMiXWQBbhQSocebYzhlkau7",
    "Llama3.2": "app-7mn8TrNM8QMgzhB1Lv51A2WB"
    }

# Ensure directories exist
os.makedirs(SYNTHETIC_TESTSET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def run_synthetic_data_generation(openai_api_key, documents_dir, num_samples, proportions):
    """
    Run the synthetic data generation process.
    """
    try:
        # Step 1: Set up API keys (OpenAI)
        setup_api_keys(openai_api_key)

        # Step 2: Configure models
        generator_llm, critic_llm, embeddings = configure_generator_models()

        # Step 3: Load documents from a dynamic directory
        documents = load_documents(documents_dir)

        # Step 4: Initialize the test set generator
        generator = initialize_generator(generator_llm, critic_llm, embeddings)

        # Step 5: Generate the test set
        df = generate_testset(generator, documents, num_samples=num_samples, proportions=proportions)

        # Step 6: Save the test set
        save_testset(df, JSON_FILE_PATH)

        # Step 7: Pass the counts to the pie chart function
        category_counts = df['evolution_type'].value_counts().to_dict()
        plot_category_distribution(category_counts, f"{SYNTHETIC_TESTSET_DIR}/evolution_type_distribution.png")

        print("Synthetic data generation completed successfully!\n")
        return True

    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
        return False


def run_evaluation(model_name, rag_model_api_key,rag_system_name):
    """
    Run the evaluation process for a specific model.
    """
    try:
        system_name=rag_system_name
        if system_name == 'ragflow':
            base_url = BASE_URL_RAGFLOW

        elif system_name == 'dify':
            base_url=BASE_URL_DIFY

        # Set the model-specific API key for OpenAI
        set_openai_api_key(OPENAI_API_KEY)

        # Ensure model-specific directory exists
        model_results_dir = f'{RESULTS_DIR}/{system_name}/{model_name}'
        os.makedirs(model_results_dir, exist_ok=True)

    
        # Prepare model-specific file paths
        generated_answers_json_path = GENERATED_ANSWERS_JSON_TEMPLATE.format(system_name=system_name, model_name=model_name)
        results_excel_path = RESULTS_EXCEL_TEMPLATE.format(system_name=system_name, model_name=model_name)
        benchmarks_json_path = BENCHMARKS_JSON_TEMPLATE.format(system_name=system_name, model_name=model_name)
        metrics_plot_path = METRICS_PLOT_TEMPLATE.format(system_name=system_name, model_name=model_name)

        # # Step 1: Load and process dataset
        df_with_answers = load_and_process_dataset(base_url, system_name, rag_model_api_key, JSON_FILE_PATH)

        # Step 2: Save dataset with generated answers
        save_generated_answers(df_with_answers, generated_answers_json_path)

        # Step 3: Load dataset for evaluation
        df_with_answers = load_dataset_for_evaluation(generated_answers_json_path)

        # Step 4: Prepare dataset for evaluation
        dataset = prepare_dataset_for_evaluation(df_with_answers)

        # Step 5: Perform evaluation
        result = evaluate_dataset(dataset)

        # Step 6: Save evaluation results
        save_evaluation_results(result, results_excel_path, benchmarks_json_path)

        # Step 7: Plot and save metrics
        plot_metrics(result, metrics_plot_path, model_name)

         # Calculate collective and category-based metrics
        metric_means = result[['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']].mean()
        simple_avg = result[result['evolution_type'] == 'simple'][['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']].mean()
        reasoning_avg = result[result['evolution_type'] == 'reasoning'][['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']].mean()
        multi_context_avg = result[result['evolution_type'] == 'multi_context'][['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']].mean()

        # Return metrics for aggregation (collective and category-based)
        return True, {
            'collective': metric_means,
            'simple': simple_avg,
            'reasoning': reasoning_avg,
            'multi_context': multi_context_avg
        }
    except Exception as e:
        print(f"Error during evaluation for {model_name}: {e}")
        return False, None



def main():
    
    # Create a log file
    log_file_name = f"experiment_log.txt"
    log_file_path = os.path.join(LOGS_DIR, log_file_name)
    with open(log_file_path, 'w') as f:
        f.write(f"Log file created at {datetime.now().strftime('%H:%M, %d/%m/%Y')}\n")

    # Start time
    tik = time.time()
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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the full pipeline to generate and evaluate synthetic data...")
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate (default: 20)')
    parser.add_argument('--proportion_simple', type=float, default=0.34, help='Proportion of simple tasks (default: 0.34)')
    parser.add_argument('--proportion_reasoning', type=float, default=0.33, help='Proportion of reasoning tasks (default: 0.33)')
    parser.add_argument('--proportion_multi_context', type=float, default=0.33, help='Proportion of multi-context tasks (default: 0.33)')
    args = parser.parse_args()

    # Prepare proportion arguments for task generation
    proportions = {simple: args.proportion_simple, reasoning: args.proportion_reasoning, multi_context: args.proportion_multi_context}
    
    # Step 1: Run the synthetic data generation
    print("\nStep 1: Running Synthetic Data Generation...\n")
    synthetic_generation_success = run_synthetic_data_generation(OPENAI_API_KEY, SOURCE_DATA_PATH, args.num_samples, proportions)
    if not synthetic_generation_success:
        print("Error: Synthetic data generation failed. Exiting the pipeline.")
        return

    # # Step 2: Run evaluation for each model in the dictionary
    all_metrics = {}
    print("\nStep 2: Running Evaluation for Multiple Models...\n")
    if system_name == 'ragflow':
        models_api_dic=ragflow_models_api_keys
    elif system_name == 'dify':
        models_api_dic=dify_models_api_keys
    
    for model_name, model_api_key in models_api_dic.items():
        print(f"Running evaluation for model: {model_name}...")

        success, metrics = run_evaluation(model_name, model_api_key, system_name)

        if not success:
            print(f"Error: Evaluation for {model_name} failed. Continuing with next model...\n")
        else:
            all_metrics[model_name] = metrics
            print(f"Evaluation for {model_name} completed successfully!\n")
    
    # Step 3: Create comparison plot for all models
    if all_metrics:
        print(all_metrics)
        comparison_dir=f'{RESULTS_DIR}/{system_name}/comparison'
        os.makedirs(comparison_dir, exist_ok=True)
        compare_models_metrics(pd.DataFrame(all_metrics), comparison_dir)

   # End time
    tok = time.time()

    # Calculate total time
    total_time = tok - tik
    print(f"\nPipeline completed successfully in {total_time:.2f} seconds!")


if __name__ == "__main__":
    main()
    # CMD: python run_pipeline.py --num_samples 1 2>&1 | tee logs/experiment_log.txt

