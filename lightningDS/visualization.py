"""
Visualization module for Lightning Network data.
This module provides functions for creating various visualizations and charts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import networkx as nx
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256, Category20
from bokeh.transform import linear_cmap
import io
import base64


def set_plotting_style():
    """Set consistent style for matplotlib plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def plot_node_distribution(nodes_df: pd.DataFrame, column: str,
                          bins: int = 30, title: Optional[str] = None) -> plt.Figure:
    """
    Create a histogram showing the distribution of a node attribute.

    Args:
        nodes_df: DataFrame containing node data
        column: Column name to plot
        bins: Number of bins for histogram
        title: Plot title (defaults to column name if None)

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots()

    if column not in nodes_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Create histogram
    sns.histplot(nodes_df[column].dropna(), bins=bins, kde=True, ax=ax)

    # Set title and labels
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    # Add statistics
    stats_text = (
        f"Mean: {nodes_df[column].mean():.2f}\n"
        f"Median: {nodes_df[column].median():.2f}\n"
        f"Std Dev: {nodes_df[column].std():.2f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_channel_capacity_distribution(channels_df: pd.DataFrame,
                                      log_scale: bool = True) -> plt.Figure:
    """
    Create a histogram showing the distribution of channel capacities.

    Args:
        channels_df: DataFrame containing channel data
        log_scale: Whether to use log scale for x-axis

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots()

    if 'capacity' not in channels_df.columns:
        raise ValueError("Column 'capacity' not found in DataFrame")

    # Create histogram
    sns.histplot(channels_df['capacity'].dropna(), bins=50, kde=False, ax=ax)

    # Set title and labels
    ax.set_title("Distribution of Channel Capacities")
    ax.set_xlabel("Capacity (sats)")
    ax.set_ylabel("Frequency")

    # Apply log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel("Capacity (sats, log scale)")

    # Add statistics
    stats_text = (
        f"Mean: {channels_df['capacity'].mean():.2f}\n"
        f"Median: {channels_df['capacity'].median():.2f}\n"
        f"Total: {channels_df['capacity'].sum():.2f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> plt.Figure:
    """
    Create a heatmap showing correlations between selected columns.

    Args:
        df: DataFrame containing data
        columns: List of column names to include in correlation matrix

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to only include numeric columns that exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]

    if not valid_columns:
        raise ValueError("No valid numeric columns found for correlation analysis")

    # Calculate correlation matrix
    corr_matrix = df[valid_columns].corr()

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str,
                    title: Optional[str] = None) -> plt.Figure:
    """
    Create a time series plot.

    Args:
        df: DataFrame containing time series data
        date_column: Name of the column containing dates
        value_column: Name of the column containing values to plot
        title: Plot title (defaults to value_column if None)

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots()

    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date
    df = df.sort_values(by=date_column)

    # Plot time series
    ax.plot(df[date_column], df[value_column], marker='o', linestyle='-', markersize=4)

    # Set title and labels
    ax.set_title(title or f"{value_column} over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_column)

    # Format x-axis
    plt.xticks(rotation=45)
    fig.autofmt_xdate()

    # Add trend line
    z = np.polyfit(range(len(df)), df[value_column], 1)
    p = np.poly1d(z)
    ax.plot(df[date_column], p(range(len(df))), "r--", alpha=0.8)

    plt.tight_layout()
    return fig


def plot_network_graph(G: nx.Graph, node_size_attr: Optional[str] = None,
                      node_color_attr: Optional[str] = None,
                      edge_width_attr: Optional[str] = None,
                      title: str = "Lightning Network Graph",
                      layout: str = "spring",
                      max_nodes: int = 500) -> plt.Figure:
    """
    Create a network visualization of the Lightning Network.

    Args:
        G: NetworkX graph representing the Lightning Network
        node_size_attr: Node attribute to use for sizing nodes (None for uniform size)
        node_color_attr: Node attribute to use for coloring nodes (None for uniform color)
        edge_width_attr: Edge attribute to use for edge widths (None for uniform width)
        title: Plot title
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        max_nodes: Maximum number of nodes to display (for performance)

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))

    # Limit graph size for performance
    if len(G) > max_nodes:
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node sizes
    if node_size_attr and node_size_attr in nx.get_node_attributes(G, node_size_attr):
        node_sizes = [G.nodes[n].get(node_size_attr, 100) for n in G.nodes()]
        # Normalize sizes
        node_sizes = [50 + (s / max(node_sizes) * 500) for s in node_sizes]
    else:
        node_sizes = 100

    # Node colors
    if node_color_attr and node_color_attr in nx.get_node_attributes(G, node_color_attr):
        node_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes()]
    else:
        node_colors = 'skyblue'

    # Edge widths
    if edge_width_attr and edge_width_attr in nx.get_edge_attributes(G, edge_width_attr):
        edge_widths = [G[u][v].get(edge_width_attr, 1) for u, v in G.edges()]
        # Normalize widths
        edge_widths = [0.5 + (w / max(edge_widths) * 3) for w in edge_widths]
    else:
        edge_widths = 1

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.8, cmap=plt.cm.viridis, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)

    # Add labels for top 10 nodes by degree
    if len(G) > 10:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
        labels = {node: node for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    else:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_interactive_network(G: nx.Graph, node_size_attr: Optional[str] = None,
                           node_color_attr: Optional[str] = None,
                           title: str = "Interactive Lightning Network Graph",
                           max_nodes: int = 300) -> None:
    """
    Create an interactive network visualization using Bokeh.

    Args:
        G: NetworkX graph representing the Lightning Network
        node_size_attr: Node attribute to use for sizing nodes
        node_color_attr: Node attribute to use for coloring nodes
        title: Plot title
        max_nodes: Maximum number of nodes to display (for performance)

    Returns:
        None (displays the plot or returns HTML)
    """
    # Limit graph size for performance
    if len(G) > max_nodes:
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)

    # Get layout
    pos = nx.spring_layout(G, seed=42)

    # Create node data
    node_data = {
        'x': [],
        'y': [],
        'name': [],
        'node_size': [],
        'node_color': [],
        'degree': []
    }

    for node in G.nodes():
        node_data['x'].append(pos[node][0])
        node_data['y'].append(pos[node][1])

        # Get node attributes
        node_attrs = G.nodes[node]
        node_data['name'].append(node_attrs.get('alias', node))

        # Node size
        if node_size_attr and node_size_attr in node_attrs:
            node_data['node_size'].append(node_attrs[node_size_attr])
        else:
            node_data['node_size'].append(G.degree(node))

        # Node color
        if node_color_attr and node_color_attr in node_attrs:
            node_data['node_color'].append(node_attrs[node_color_attr])
        else:
            node_data['node_color'].append(G.degree(node))

        node_data['degree'].append(G.degree(node))

    # Create edge data
    edge_data = {
        'x0': [],
        'y0': [],
        'x1': [],
        'y1': []
    }

    for edge in G.edges():
        edge_data['x0'].append(pos[edge[0]][0])
        edge_data['y0'].append(pos[edge[0]][1])
        edge_data['x1'].append(pos[edge[1]][0])
        edge_data['y1'].append(pos[edge[1]][1])

    # Create Bokeh figure
    p = figure(title=title, width=800, height=600, tools="pan,wheel_zoom,box_zoom,reset,save")

    # Add edges
    p.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=edge_data,
             line_width=0.5, line_color='gray', line_alpha=0.3)

    # Add nodes
    source = ColumnDataSource(node_data)

    # Map node size
    size_mapper = linear_cmap(field_name='node_size', palette=Viridis256,
                             low=min(node_data['node_size']), high=max(node_data['node_size']))

    # Create scatter plot for nodes
    r = p.circle('x', 'y', size=10, source=source,
                fill_color=size_mapper, line_color='black', line_width=0.5,
                alpha=0.8)

    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Name", "@name"),
        ("Degree", "@degree"),
    ])
    p.add_tools(hover)

    # Add color bar
    color_bar = ColorBar(color_mapper=size_mapper['transform'], width=8, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # Remove grid and axes
    p.grid.grid_line_color = None
    p.axis.visible = False

    # Return the plot
    return p


def plot_community_graph(G: nx.Graph, communities: Dict[str, int],
                        title: str = "Lightning Network Communities",
                        max_nodes: int = 500) -> plt.Figure:
    """
    Create a network visualization with nodes colored by community.

    Args:
        G: NetworkX graph representing the Lightning Network
        communities: Dictionary mapping node IDs to community IDs
        title: Plot title
        max_nodes: Maximum number of nodes to display (for performance)

    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))

    # Limit graph size for performance
    if len(G) > max_nodes:
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)

    # Get layout
    pos = nx.spring_layout(G, seed=42)

    # Get community colors
    unique_communities = set(communities.values())
    color_map = plt.cm.get_cmap('tab20', len(unique_communities))

    # Draw nodes colored by community
    for i, comm in enumerate(unique_communities):
        # Get nodes in this community
        comm_nodes = [node for node in G.nodes() if communities.get(node, -1) == comm]
        nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes, node_color=[color_map(i)],
                              node_size=100, alpha=0.8, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, ax=ax)

    # Add labels for top nodes in each community
    labels = {}
    for comm in unique_communities:
        comm_nodes = [node for node in G.nodes() if communities.get(node, -1) == comm]
        if comm_nodes:
            # Get top node by degree
            top_node = max(comm_nodes, key=lambda n: G.degree(n))
            labels[top_node] = f"Comm {comm}"

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')

    # Add legend for communities
    handles = []
    for i, comm in enumerate(unique_communities):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i),
                                 markersize=10, label=f'Community {comm}'))

    ax.legend(handles=handles, loc='upper right', title="Communities")

    plt.tight_layout()
    return fig


def plot_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.

    Args:
        fig: Matplotlib figure object

    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def create_dashboard(nodes_df: pd.DataFrame, channels_df: pd.DataFrame,
                    output_path: str = "lightning_dashboard.html") -> str:
    """
    Create an HTML dashboard with multiple visualizations.

    Args:
        nodes_df: DataFrame containing node data
        channels_df: DataFrame containing channel data
        output_path: Path to save the HTML file

    Returns:
        Path to the saved HTML file
    """
    # Create visualizations
    figs = []

    # Node distribution
    if 'channel_count' in nodes_df.columns:
        figs.append(plot_node_distribution(nodes_df, 'channel_count',
                                         title="Distribution of Channels per Node"))

    # Channel capacity distribution
    if 'capacity' in channels_df.columns:
        figs.append(plot_channel_capacity_distribution(channels_df))

    # Convert figures to base64
    img_strs = [plot_to_base64(fig) for fig in figs]

    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lightning Network Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .dashboard { display: flex; flex-wrap: wrap; }
            .chart { margin: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); padding: 10px; }
            h1 { color: #333; }
            .stats { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Lightning Network Dashboard</h1>

        <div class="stats">
            <h2>Network Statistics</h2>
            <p>Total Nodes: {total_nodes}</p>
            <p>Total Channels: {total_channels}</p>
            <p>Total Capacity: {total_capacity} sats</p>
        </div>

        <div class="dashboard">
    """.format(
        total_nodes=len(nodes_df),
        total_channels=len(channels_df),
        total_capacity=channels_df['capacity'].sum() if 'capacity' in channels_df.columns else 'N/A'
    )

    # Add images
    for i, img_str in enumerate(img_strs):
        html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{img_str}" alt="Chart {i+1}">
        </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path