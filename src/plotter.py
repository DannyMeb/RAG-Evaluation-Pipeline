# Project: Rag systems evaluation pipeline using Ragas
# Authored by Daniel
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_metrics(result_df, plot_path, model_name):
    metrics_columns = ['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']

    if all(column in result_df.columns for column in metrics_columns):
        # Get collective average for all metrics
        collective_avg = result_df[metrics_columns].mean()

        # Get category-specific averages
        simple_avg = result_df[result_df['evolution_type'] == 'simple'][metrics_columns].mean()
        reasoning_avg = result_df[result_df['evolution_type'] == 'reasoning'][metrics_columns].mean()
        multi_context_avg = result_df[result_df['evolution_type'] == 'multi_context'][metrics_columns].mean()

        # Create a DataFrame to hold all averages
        category_averages = pd.DataFrame({
            'Collective': collective_avg,
            'Simple': simple_avg,
            'Reasoning': reasoning_avg,
            'Multi_context': multi_context_avg
        })

        # Transpose the DataFrame for easier plotting (metrics as rows, categories as columns)
        category_averages = category_averages.T

        # Set up the bar plot with calm and simple colors
        x = np.arange(len(metrics_columns))  # Label locations for metrics
        width = 0.2  # The width of the bars

        # Updated colors to be softer and simple
        colors = ['#6D98BA', '#A8D5BA', '#FFDDC1', '#FFE699']  # Calm blue, green, beige, yellow

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each category
        rects1 = ax.bar(x - width * 1.5, category_averages.loc['Collective'], width, label='Collective', color=colors[0])
        rects2 = ax.bar(x - width / 2, category_averages.loc['Simple'], width, label='Simple', color=colors[1])
        rects3 = ax.bar(x + width / 2, category_averages.loc['Reasoning'], width, label='Reasoning', color=colors[2])
        rects4 = ax.bar(x + width * 1.5, category_averages.loc['Multi_context'], width, label='Multi_context', color=colors[3])

        # Add some text for labels, title, and custom x-axis tick labels
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title(f'Average Metrics Evaluation Scores by Category for {model_name}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_columns, fontsize=12)
        ax.set_ylim(0, 1)  # Since metrics range from 0 to 1
        ax.legend()

        # Annotate each bar with the value
        def annotate_bars(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',  # Format height to 2 decimal places
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        annotate_bars(rects1)
        annotate_bars(rects2)
        annotate_bars(rects3)
        annotate_bars(rects4)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        print(f"Metrics plot for {model_name} saved at {plot_path}")

def compare_models_metrics(metrics_dict, dir):
    """
    Plot a comparison of evaluation metrics across all models for collective, simple, reasoning, and multi_context.
    """
    # Metrics to compare
    metrics_columns = ['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']
    
    # Create a list for the category types to loop through
    categories = ['collective', 'simple', 'reasoning', 'multi_context']
    
    # Use calm and simple colors
    colors = ['#6D98BA', '#A8D5BA', '#FFDDC1', '#FFE699']  # Calm blue, green, beige, yellow
    
    for category in categories:
        # Collect the metrics for each model for the current category
        category_metrics = {model: metrics[category] for model, metrics in metrics_dict.items()}
        
        # Convert to DataFrame for easier plotting (metrics as columns, models as rows)
        metrics_df = pd.DataFrame(category_metrics).T[metrics_columns]
        
        # Handle NaN by filling with 0.0 (you can change this depending on your use case)
        metrics_df = metrics_df.fillna(0)
        
        # Set up the figure and axes for the current category
        metrics = metrics_df.columns  # Metric names on x-axis
        models = metrics_df.index  # Model names for grouping
        x = np.arange(len(metrics))  # Label locations for metrics on x-axis
        width = 0.2  # The width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bars for each model using the soft colors
        for i, model in enumerate(models):
            rects = ax.bar(x + i * width, metrics_df.loc[model], width, label=model, color=colors[i % len(colors)])
            # Attach a text label above each bar, displaying its height (value)
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',  # Format the height value to 2 decimal places
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Add labels, title, and legend
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title(f'Comparison of {category.capitalize()} Evaluation Metrics Across Models', fontsize=16)
        ax.set_xticks(x + width)  # Center the labels between grouped bars
        ax.set_xticklabels(metrics)  # Set metric names as labels on x-axis
        ax.set_ylim(0, 1)  # Set y-axis limit since scores are between 0 and 1
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot for the current category
        comparison_plot_path = f'{dir}/metrics_comparison_{category}.png'
        plt.savefig(comparison_plot_path)
        plt.close()

        print(f"Comparison plot for {category} saved at {comparison_plot_path}")



def plot_category_distribution(category_counts, plot_path):
    """
    Create a pie chart to visualize the proportions of categories (simple, reasoning, multi_context).
    """
    # Prepare labels and sizes for the pie chart
    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    colors = ['#4CAF50', '#FF9800', '#2196F3']  # Simple colors for the pie chart

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add a title to the pie chart
    ax.set_title('Category Distribution')

    # Save the plot to the specified path
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Pie chart saved at {plot_path}")
