"""
Analysis module for Lightning Network data.
This module provides advanced analytical functions for deeper insights.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def create_network_graph(nodes_df: pd.DataFrame, channels_df: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph from node and channel data.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data

    Returns:
        NetworkX graph representing the Lightning Network
    """
    G = nx.Graph()

    # Add nodes
    for _, node in nodes_df.iterrows():
        node_attrs = node.to_dict()
        G.add_node(node_attrs.get('pub_key', ''), **node_attrs)

    # Add edges (channels)
    for _, channel in channels_df.iterrows():
        if 'node1_pub' in channel and 'node2_pub' in channel:
            edge_attrs = channel.to_dict()
            G.add_edge(channel['node1_pub'], channel['node2_pub'], **edge_attrs)

    return G


def identify_central_nodes(G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """
    Identify the most central nodes in the network using various centrality measures.

    Args:
        G: NetworkX graph representing the Lightning Network
        top_n: Number of top nodes to return for each measure

    Returns:
        DataFrame with top nodes for each centrality measure
    """
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Convert to DataFrames
    df_degree = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['degree_centrality'])
    df_betweenness = pd.DataFrame.from_dict(betweenness_centrality, orient='index', columns=['betweenness_centrality'])
    df_closeness = pd.DataFrame.from_dict(closeness_centrality, orient='index', columns=['closeness_centrality'])

    # Combine all measures
    df_centrality = df_degree.join(df_betweenness).join(df_closeness)
    df_centrality.index.name = 'pub_key'
    df_centrality = df_centrality.reset_index()

    # Get top nodes for each measure
    top_degree = df_centrality.nlargest(top_n, 'degree_centrality')
    top_betweenness = df_centrality.nlargest(top_n, 'betweenness_centrality')
    top_closeness = df_centrality.nlargest(top_n, 'closeness_centrality')

    return {
        'degree': top_degree,
        'betweenness': top_betweenness,
        'closeness': top_closeness,
        'combined': df_centrality
    }


def find_communities(G: nx.Graph, method: str = 'louvain') -> Dict[str, Any]:
    """
    Detect communities in the Lightning Network.

    Args:
        G: NetworkX graph representing the Lightning Network
        method: Community detection method ('louvain', 'label_propagation', or 'greedy_modularity')

    Returns:
        Dictionary with community assignments and statistics
    """
    if method == 'louvain':
        try:
            from community import best_partition
            partition = best_partition(G)
        except ImportError:
            raise ImportError("Please install python-louvain package: pip install python-louvain")
    elif method == 'label_propagation':
        partition = {node: i for i, comm in enumerate(nx.algorithms.community.label_propagation_communities(G))
                    for node in comm}
    elif method == 'greedy_modularity':
        partition = {node: i for i, comm in enumerate(nx.algorithms.community.greedy_modularity_communities(G))
                    for node in comm}
    else:
        raise ValueError(f"Unknown community detection method: {method}")

    # Convert to DataFrame
    community_df = pd.DataFrame.from_dict(partition, orient='index', columns=['community'])
    community_df.index.name = 'pub_key'
    community_df = community_df.reset_index()

    # Get community sizes
    community_sizes = community_df['community'].value_counts().sort_values(ascending=False)

    # Calculate modularity
    modularity = nx.algorithms.community.modularity(G,
                                                   [set(community_df[community_df['community'] == c]['pub_key'])
                                                    for c in community_df['community'].unique()])

    return {
        'assignments': community_df,
        'sizes': community_sizes,
        'modularity': modularity,
        'num_communities': len(community_sizes)
    }


def cluster_nodes(nodes_df: pd.DataFrame, features: List[str],
                 method: str = 'kmeans', n_clusters: int = 5) -> pd.DataFrame:
    """
    Cluster nodes based on specified features.

    Args:
        nodes_df: DataFrame containing node data
        features: List of features to use for clustering
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for KMeans

    Returns:
        DataFrame with original data and cluster assignments
    """
    # Check if all features exist in the DataFrame
    missing_features = [f for f in features if f not in nodes_df.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")

    # Select and scale features
    X = nodes_df[features].copy()

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = clusterer.fit_predict(X_scaled)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        clusters = clusterer.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Add cluster assignments to original data
    result = nodes_df.copy()
    result['cluster'] = clusters

    # Calculate cluster statistics
    cluster_stats = result.groupby('cluster')[features].mean()

    return {
        'clustered_data': result,
        'cluster_stats': cluster_stats,
        'n_clusters': len(cluster_stats)
    }


def path_analysis(G: nx.Graph, source_nodes: List[str], target_nodes: List[str]) -> Dict[str, Any]:
    """
    Analyze paths between source and target nodes.

    Args:
        G: NetworkX graph representing the Lightning Network
        source_nodes: List of source node public keys
        target_nodes: List of target node public keys

    Returns:
        Dictionary with path statistics
    """
    results = {
        'paths': [],
        'avg_path_length': 0,
        'max_path_length': 0,
        'min_path_length': float('inf'),
        'path_length_distribution': {}
    }

    path_lengths = []

    # Find paths between each source-target pair
    for source in source_nodes:
        for target in target_nodes:
            if source == target:
                continue

            if source in G and target in G:
                try:
                    path = nx.shortest_path(G, source=source, target=target)
                    path_length = len(path) - 1  # Number of edges

                    results['paths'].append({
                        'source': source,
                        'target': target,
                        'path': path,
                        'length': path_length
                    })

                    path_lengths.append(path_length)
                except nx.NetworkXNoPath:
                    # No path exists
                    pass

    # Calculate statistics
    if path_lengths:
        results['avg_path_length'] = np.mean(path_lengths)
        results['max_path_length'] = max(path_lengths)
        results['min_path_length'] = min(path_lengths)

        # Path length distribution
        for length in path_lengths:
            results['path_length_distribution'][length] = results['path_length_distribution'].get(length, 0) + 1

    return results


def time_series_analysis(df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
    """
    Perform time series analysis on Lightning Network data.

    Args:
        df: DataFrame containing time series data
        date_column: Name of the column containing dates
        value_column: Name of the column containing values to analyze

    Returns:
        Dictionary with time series analysis results
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Set date as index
    ts_df = df.set_index(date_column)

    # Resample to different frequencies
    daily = ts_df[value_column].resample('D').mean()
    weekly = ts_df[value_column].resample('W').mean()
    monthly = ts_df[value_column].resample('M').mean()

    # Calculate growth rates
    daily_pct_change = daily.pct_change().dropna()
    weekly_pct_change = weekly.pct_change().dropna()
    monthly_pct_change = monthly.pct_change().dropna()

    # Calculate statistics
    stats = {
        'daily': {
            'mean': daily.mean(),
            'std': daily.std(),
            'min': daily.min(),
            'max': daily.max(),
            'growth_rate': daily_pct_change.mean()
        },
        'weekly': {
            'mean': weekly.mean(),
            'std': weekly.std(),
            'min': weekly.min(),
            'max': weekly.max(),
            'growth_rate': weekly_pct_change.mean()
        },
        'monthly': {
            'mean': monthly.mean(),
            'std': monthly.std(),
            'min': monthly.min(),
            'max': monthly.max(),
            'growth_rate': monthly_pct_change.mean()
        }
    }

    # Detect trend
    from scipy import stats as scipy_stats

    # Linear regression for trend
    x = np.arange(len(monthly))
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, monthly)

    trend_analysis = {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend': 'increasing' if slope > 0 else 'decreasing'
    }

    return {
        'resampled_data': {
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly
        },
        'statistics': stats,
        'trend_analysis': trend_analysis
    }