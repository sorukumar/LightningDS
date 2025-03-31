"""
Data ingestion module for Lightning Network data.
This module handles loading data from various sources including JSON files and converting to CSV.
"""
import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON in file: {file_path}")


def json_to_csv(json_file_path: str, output_dir: str, prefix: str = "") -> Tuple[str, str]:
    """
    Convert Lightning Network JSON data to CSV files for nodes and channels.

    Args:
        json_file_path: Path to the JSON file
        output_dir: Directory to save the CSV files
        prefix: Optional prefix for output filenames

    Returns:
        Tuple of (nodes_csv_path, channels_csv_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    data = load_json_data(json_file_path)

    # Extract nodes and channels
    nodes = data.get('nodes', [])
    channels = data.get('edges', [])

    # Convert to DataFrames
    nodes_df = pd.DataFrame(nodes)
    channels_df = pd.DataFrame(channels)

    # Define output paths
    nodes_csv_path = os.path.join(output_dir, f"{prefix}nodes.csv")
    channels_csv_path = os.path.join(output_dir, f"{prefix}channels.csv")

    # Save to CSV
    nodes_df.to_csv(nodes_csv_path, index=False)
    channels_df.to_csv(channels_csv_path, index=False)

    return nodes_csv_path, channels_csv_path


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the CSV data

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def load_multiple_csv_files(directory: str, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files from a directory.

    Args:
        directory: Directory containing CSV files
        pattern: Glob pattern to match files

    Returns:
        Dictionary mapping filenames to DataFrames
    """
    import glob

    data_frames = {}
    for file_path in glob.glob(os.path.join(directory, pattern)):
        file_name = os.path.basename(file_path)
        data_frames[file_name] = load_csv_data(file_path)

    return data_frames