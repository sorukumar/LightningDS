"""
Transformation module for Lightning Network data.
This module handles cleaning, transforming, and preparing data for analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union


def clean_node_data(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare node data for analysis.

    Args:
        nodes_df: DataFrame containing node data

    Returns:
        Cleaned DataFrame with consistent data types and missing values handled
    """
    df = nodes_df.copy()

    # Handle missing values
    if 'last_update' in df.columns:
        df['last_update'] = pd.to_datetime(df['last_update'], unit='s', errors='coerce')

    # Convert capacity to numeric if present
    if 'capacity' in df.columns:
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')

    # Handle aliases - ensure they're strings
    if 'alias' in df.columns:
        df['alias'] = df['alias'].fillna('').astype(str)

    return df


def clean_channel_data(channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare channel data for analysis.

    Args:
        channels_df: DataFrame containing channel data

    Returns:
        Cleaned DataFrame with consistent data types and missing values handled
    """
    df = channels_df.copy()

    # Handle missing values and convert data types
    if 'last_update' in df.columns:
        df['last_update'] = pd.to_datetime(df['last_update'], unit='s', errors='coerce')

    # Convert capacity to numeric
    if 'capacity' in df.columns:
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')

    # Ensure node identifiers are strings
    if 'node1_pub' in df.columns:
        df['node1_pub'] = df['node1_pub'].astype(str)
    if 'node2_pub' in df.columns:
        df['node2_pub'] = df['node2_pub'].astype(str)

    return df


def categorize_nodes(nodes_df: pd.DataFrame, capacity_threshold: int = 1000000) -> pd.DataFrame:
    """
    Categorize nodes based on their capacity and other attributes.

    Args:
        nodes_df: DataFrame containing node data
        capacity_threshold: Threshold for categorizing high-capacity nodes (in sats)

    Returns:
        DataFrame with added category column
    """
    df = nodes_df.copy()

    # Create category column
    if 'capacity' in df.columns:
        df['category'] = 'standard'
        df.loc[df['capacity'] > capacity_threshold, 'category'] = 'high_capacity'

    # Additional categorization based on other attributes could be added here

    return df


def merge_node_channel_data(nodes_df: pd.DataFrame, channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge node and channel data for comprehensive analysis.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data

    Returns:
        Merged DataFrame with node and channel information
    """
    # Ensure we have the necessary columns
    if 'pub_key' not in nodes_df.columns or 'node1_pub' not in channels_df.columns:
        raise ValueError("Required columns missing for merge operation")

    # Create a copy to avoid modifying the original
    channels = channels_df.copy()

    # Merge node1 information
    merged = channels.merge(
        nodes_df,
        left_on='node1_pub',
        right_on='pub_key',
        how='left',
        suffixes=('', '_node1')
    )

    # Merge node2 information
    merged = merged.merge(
        nodes_df,
        left_on='node2_pub',
        right_on='pub_key',
        how='left',
        suffixes=('', '_node2')
    )

    return merged


def create_node_metrics(nodes_df: pd.DataFrame, channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional metrics for nodes based on their channels.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data

    Returns:
        DataFrame with additional calculated metrics
    """
    df = nodes_df.copy()

    # Count channels per node
    if 'pub_key' in df.columns and 'node1_pub' in channels_df.columns and 'node2_pub' in channels_df.columns:
        # Count channels where node is node1
        node1_counts = channels_df['node1_pub'].value_counts().to_dict()
        # Count channels where node is node2
        node2_counts = channels_df['node2_pub'].value_counts().to_dict()

        # Combine counts
        channel_counts = {}
        for node, count in node1_counts.items():
            channel_counts[node] = channel_counts.get(node, 0) + count
        for node, count in node2_counts.items():
            channel_counts[node] = channel_counts.get(node, 0) + count

        # Add to dataframe
        df['channel_count'] = df['pub_key'].map(channel_counts).fillna(0).astype(int)

        # Calculate total capacity
        node_capacities = {}
        for _, row in channels_df.iterrows():
            if 'capacity' in row and 'node1_pub' in row and 'node2_pub' in row:
                capacity = pd.to_numeric(row['capacity'], errors='coerce')
                if not pd.isna(capacity):
                    node1 = row['node1_pub']
                    node2 = row['node2_pub']
                    node_capacities[node1] = node_capacities.get(node1, 0) + capacity
                    node_capacities[node2] = node_capacities.get(node2, 0) + capacity

        df['total_capacity'] = df['pub_key'].map(node_capacities).fillna(0)

    return df


def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame to a 0-1 scale.

    Args:
        df: Input DataFrame
        columns: List of columns to normalize

    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()

    for column in columns:
        if column in result.columns:
            min_val = result[column].min()
            max_val = result[column].max()

            if max_val > min_val:  # Avoid division by zero
                result[column] = (result[column] - min_val) / (max_val - min_val)

    return result


def create_time_series(df: pd.DataFrame, date_column: str, value_column: str,
                       freq: str = 'D') -> pd.DataFrame:
    """
    Create a time series DataFrame from a date column and value column.

    Args:
        df: Input DataFrame
        date_column: Name of the column containing dates
        value_column: Name of the column containing values
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)

    Returns:
        Time series DataFrame
    """
    # Ensure date column is datetime
    if date_column in df.columns and value_column in df.columns:
        ts_df = df.copy()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])

        # Set date as index and resample
        ts_df = ts_df.set_index(date_column)
        resampled = ts_df[value_column].resample(freq).mean()

        return resampled.reset_index()
    else:
        raise ValueError(f"Columns {date_column} or {value_column} not found in DataFrame")