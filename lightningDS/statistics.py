"""
Statistics module for Lightning Network data.
This module provides functions for calculating descriptive statistics and network metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple


def basic_node_stats(nodes_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for node data.

    Args:
        nodes_df: DataFrame containing node data

    Returns:
        Dictionary of statistics including counts, averages, and distributions
    """
    stats = {}

    # Basic counts
    stats['total_nodes'] = len(nodes_df)

    # Capacity statistics if available
    if 'capacity' in nodes_df.columns:
        capacity_stats = nodes_df['capacity'].describe()
        stats['capacity'] = {
            'mean': capacity_stats['mean'],
            'median': nodes_df['capacity'].median(),
            'min': capacity_stats['min'],
            'max': capacity_stats['max'],
            'std': capacity_stats['std']
        }

    # Channel count statistics if available
    if 'channel_count' in nodes_df.columns:
        channel_stats = nodes_df['channel_count'].describe()
        stats['channel_count'] = {
            'mean': channel_stats['mean'],
            'median': nodes_df['channel_count'].median(),
            'min': channel_stats['min'],
            'max': channel_stats['max'],
            'std': channel_stats['std']
        }

        # Distribution of nodes by channel count
        channel_dist = nodes_df['channel_count'].value_counts().sort_index()
        stats['channel_distribution'] = channel_dist.to_dict()

    return stats


def basic_channel_stats(channels_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for channel data.

    Args:
        channels_df: DataFrame containing channel data

    Returns:
        Dictionary of statistics including counts, capacities, and age distribution
    """
    stats = {}

    # Basic counts
    stats['total_channels'] = len(channels_df)

    # Capacity statistics
    if 'capacity' in channels_df.columns:
        capacity_stats = channels_df['capacity'].describe()
        stats['capacity'] = {
            'mean': capacity_stats['mean'],
            'median': channels_df['capacity'].median(),
            'min': capacity_stats['min'],
            'max': capacity_stats['max'],
            'std': capacity_stats['std'],
            'total': channels_df['capacity'].sum()
        }

        # Distribution of channels by capacity range
        stats['capacity_distribution'] = {}
        capacity_ranges = [
            (0, 100000),
            (100000, 1000000),
            (1000000, 5000000),
            (5000000, 10000000),
            (10000000, float('inf'))
        ]

        for low, high in capacity_ranges:
            if high == float('inf'):
                label = f"{low/1000000:.1f}M+"
            else:
                label = f"{low/1000000:.1f}M-{high/1000000:.1f}M"

            count = ((channels_df['capacity'] >= low) &
                     (channels_df['capacity'] < high)).sum()
            stats['capacity_distribution'][label] = count

    # Age statistics if last_update is available
    if 'last_update' in channels_df.columns:
        channels_df['last_update'] = pd.to_datetime(channels_df['last_update'], errors='coerce')
        if not channels_df['last_update'].isna().all():
            current_time = pd.Timestamp.now()
            channels_df['age_days'] = (current_time - channels_df['last_update']).dt.days

            age_stats = channels_df['age_days'].describe()
            stats['age_days'] = {
                'mean': age_stats['mean'],
                'median': channels_df['age_days'].median(),
                'min': age_stats['min'],
                'max': age_stats['max'],
                'std': age_stats['std']
            }

    return stats


def network_density(nodes_df: pd.DataFrame, channels_df: pd.DataFrame) -> float:
    """
    Calculate the density of the Lightning Network.

    Network density is the ratio of actual connections to potential connections.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data

    Returns:
        Network density as a float between 0 and 1
    """
    n_nodes = len(nodes_df)
    n_channels = len(channels_df)

    # Maximum possible edges in an undirected graph
    max_possible_edges = (n_nodes * (n_nodes - 1)) / 2

    if max_possible_edges > 0:
        return n_channels / max_possible_edges
    else:
        return 0.0


def node_degree_distribution(channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the degree distribution of nodes in the network.

    Args:
        channels_df: DataFrame containing channel data with node1_pub and node2_pub columns

    Returns:
        DataFrame with node degrees and their frequencies
    """
    if 'node1_pub' not in channels_df.columns or 'node2_pub' not in channels_df.columns:
        raise ValueError("Channel data must contain node1_pub and node2_pub columns")

    # Count occurrences of each node
    node1_counts = channels_df['node1_pub'].value_counts()
    node2_counts = channels_df['node2_pub'].value_counts()

    # Combine counts
    all_nodes = pd.concat([node1_counts, node2_counts], axis=1)
    all_nodes = all_nodes.fillna(0)
    all_nodes['degree'] = all_nodes.sum(axis=1).astype(int)

    # Get distribution
    degree_dist = all_nodes['degree'].value_counts().sort_index().reset_index()
    degree_dist.columns = ['degree', 'frequency']

    return degree_dist


def calculate_centrality(nodes_df: pd.DataFrame, channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic centrality metrics for nodes.

    This function calculates degree centrality, which is the number of connections
    a node has divided by the maximum possible connections.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data

    Returns:
        DataFrame with nodes and their centrality metrics
    """
    if 'pub_key' not in nodes_df.columns:
        raise ValueError("Node data must contain pub_key column")

    if 'node1_pub' not in channels_df.columns or 'node2_pub' not in channels_df.columns:
        raise ValueError("Channel data must contain node1_pub and node2_pub columns")

    # Count occurrences of each node
    node1_counts = channels_df['node1_pub'].value_counts()
    node2_counts = channels_df['node2_pub'].value_counts()

    # Combine counts to get degree
    degree_counts = pd.DataFrame({
        'pub_key': pd.concat([node1_counts, node2_counts], axis=0).index.unique()
    })
    degree_counts['degree'] = 0

    for node in degree_counts['pub_key']:
        count1 = node1_counts.get(node, 0)
        count2 = node2_counts.get(node, 0)
        degree_counts.loc[degree_counts['pub_key'] == node, 'degree'] = count1 + count2

    # Calculate degree centrality
    n_nodes = len(nodes_df)
    if n_nodes > 1:
        degree_counts['degree_centrality'] = degree_counts['degree'] / (n_nodes - 1)
    else:
        degree_counts['degree_centrality'] = 0

    # Merge with node data
    result = nodes_df.merge(degree_counts, on='pub_key', how='left')
    result['degree'] = result['degree'].fillna(0).astype(int)
    result['degree_centrality'] = result['degree_centrality'].fillna(0)

    return result


def correlation_analysis(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.

    Args:
        df: DataFrame containing data
        columns: List of column names to include in correlation analysis

    Returns:
        Correlation matrix as a DataFrame
    """
    # Filter to only include numeric columns that exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]

    if not valid_columns:
        raise ValueError("No valid numeric columns found for correlation analysis")

    return df[valid_columns].corr()