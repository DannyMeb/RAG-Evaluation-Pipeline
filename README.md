# Rag systems evaluation pipeline using Ragas

## Introduction
The Ragas Project is designed to generate and evaluate synthetic test sets for AI models, focusing on models like Falcon and Llama. This framework supports various evaluation categories and utilizes advanced techniques to provide insights into model performance.


## Evaluation Metrics:
- Faithfullness:A faithful response should not contain hallucinations or contradictions compared to the knowledge found in the retrieved documents or external knowledge sources. 

- Context Recall:It focuses on how well the system retrieves necessary and accurate information.

- Answer Relevance:It assesses the semantic similarity between the question and the response, ensuring that the generated answer is meaningful and directly addresses the query.

- Context Precision: It assesses the specificity and accuracy of the retrieval process.

## [more evaluations](https://docs.ragas.io/en/stable/concepts/metrics/)
## Project Structure

The project directory contains several key components:

- `src/`: Contains all the source code.
  - `evaluate.py`: Module for evaluating test sets.
  - `generator.py`: Handles the generation of synthetic test sets.
  - `plotter.py`: Provides plotting functions for visualization of results.
- `scripts/`: Contains scripts for environment setup.
  - `create_conda_env.sh`: Script to create and activate a Conda environment.
- `data/`: Directory for storing input data for the models.
- `results/`: Output directory where evaluation results are saved.
- `README.md`: Documentation for the project.
- `requirements.txt`: Specifies the Python package dependencies.
- `run_pipeline.py`: Main executable script for running the generation and evaluation pipeline.

## Installation

### Prerequisites

- Anaconda or Miniconda (for managing Conda environments)
- Make sure to have your OpenAI API Key and update it in `run_pipeline.py`
  ```bash
  OPENAI_API_KEY = "" 

- Make sure you have Ollama instealled and running
- Make sure to have your RagFlow/Dify system is up and running
- Make sure to integrate Falcons (Falcon1-7B and Falcon2-11B) and Llama (llama3.1 and llama3.2) with your Rag system via ollama model provider. 
- Make sure to Update the IP ADDRESS of the BASE_URL in `run_pipeline.py`
   ```bash
    BASE_URL_DIFY = "http://<IP_ADDRESS>/v1/"
    BASE_URL_RAGFLOW = "http://<IP_ADDRESS>/v1/api/"

## Pipeline Overview

This pipeline processes and evaluates multiple language models (RagFlow, Dify) by generating synthetic data, processing it, and running evaluations. Here’s a step-by-step breakdown of what happens when the pipeline is run:

### Steps

**MUST READ**: You must first deploy any LLM Model (Llama, Falcon, Qwen, and so on) on your target device using Ollama. You also need to run either RagFlow or Dify (they both are rag frameworks) and add the your prefered model in the Rag framework using the end ponint API Provided by Ollama. Only then you can use this repo to evaulte your rag pipeline.
1. **Choose the System**:
    -  The user is prompted to choose between RagFlow or Dify as the system to be used for evaluation.
    - Based on the selection, the pipeline will use system-specific API keys and URLs.

2. **Load PDF Files**:
    - The pipeline loads the PDF files from the `/data` directory.
    - These files should be the same ones used to create the knowledge graph.
    
3. **Configure Models**:
    - The pipeline sets up the API keys for OpenAI.
    - Models are configured for both generation and evaluation processes (LLM, critic LLM, embeddings).

4. **Generate Synthetic Data** (Checkpoint-1):
    - The pipeline generates synthetic test data based on the documents loaded.
    - The user specifies the number of samples and the proportions for different task types (simple, reasoning, multi-context).
    - The synthetic test set is saved in the `/synthetic_testset` folder as `Synthetic_testset.json`.

5. **Save Category Distribution Plot** (Checkpoint-2):
    - A pie chart showing the distribution of task categories (simple, reasoning, multi-context) is generated and saved in the `/synthetic_testset` folder as `evolution_type_distribution.png`.

6. **Run Evaluation for Each Model**:
    - For each model in the selected system (RagFlow or Dify), the pipeline:
      - Loads the synthetic test set.
      - Sends the test set to the model API to generate answers.
      - Saves the generated answers in the `Results/<system_name>/<model_name>` folder.

7. **Save Generated Answers** (Checkpoint-3):
    - The generated answers for each model are saved in `Results/<system_name>/<model_name>` as `Synthetic_testset_with_answers_<model_name>.json`.

8. **Save Evaluation Results** (Checkpoint-4):
    - Evaluation results (including metrics) are saved in the `Results/<system_name>/<model_name>` folder:
      - Benchmarks in Excel format: `benchmarks_<model_name>.xlsx`
      - Benchmarks in JSON format: `benchmarks_<model_name>.json`
      - Metrics plot: `average_metrics_plot_<model_name>.png`

9. **Plot Comparison for All Models** (Checkpoint-5):
    - Once the evaluation for all models is completed, a comparison plot is generated, showing the performance of all models.
    - The comparison is saved in the `Results/<system_name>` folder.

10. **Log Pipeline Execution** (Checkpoint-6):
    - Throughout the pipeline, a log file is created and updated in the `/logs` directory, capturing the progress and any errors encountered during execution.

11. **Completion**:
    - Once all steps are completed, the pipeline prints a success message along with the total time taken for execution.


## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DannyMeb/RAG-Evaluation-Pipeline.git
   cd RAG-Evaluation-Framework

2. **Create and Activate the Conda Environment:**
   ```bash
   scripts/create_conda_env.sh

3. **Usaage:**
   ```bash
   conda activate ragas_env
   python run_pipeline.py --num_samples <n> --proportion_simple <x> --proportion_reasoning <y> --proportion_multi_context <z>
