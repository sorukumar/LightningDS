
# LightningDS

A Python library for analyzing Lightning Network data.

## Overview

LightningDS is a data science toolkit designed specifically for working with Lightning Network data. It provides tools for data ingestion, transformation, statistical analysis, and visualization of Lightning Network nodes and channels.

## Features

- **Data Ingestion**: Load data from JSON files and convert to CSV format
- **Data Transformation**: Clean, transform, and prepare Lightning Network data
- **Statistical Analysis**: Calculate descriptive statistics and network metrics
- **Advanced Analysis**: Perform community detection, centrality analysis, and clustering
- **Visualization**: Create insightful visualizations of network properties

## Installation

```bash
pip install lightningDS
```

Or install from source:

```bash
git clone https://github.com/sorukumar/lightningDS.git
cd lightningDS
pip install -e .
```

## Quick Start

```python
import lightningDS as lds

# Load data
data = lds.ingestion.load_json_data("lightning_network_data.json")

# Convert to CSV
nodes_csv, channels_csv = lds.ingestion.json_to_csv(
    "lightning_network_data.json", "output_directory"
)

# Load CSV data
nodes_df = lds.ingestion.load_csv_data(nodes_csv)
channels_df = lds.ingestion.load_csv_data(channels_csv)

# Clean data
nodes_df = lds.transformation.clean_node_data(nodes_df)
channels_df = lds.transformation.clean_channel_data(channels_df)

# Calculate statistics
node_stats = lds.statistics.basic_node_stats(nodes_df)
channel_stats = lds.statistics.basic_channel_stats(channels_df)

# Create network graph
import networkx as nx
G = lds.analysis.create_network_graph(nodes_df, channels_df)

# Visualize
lds.visualization.plot_node_distribution(nodes_df, "channel_count")
lds.visualization.plot_network_graph(G, node_size_attr="channel_count")
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
