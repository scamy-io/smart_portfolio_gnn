# Data Dictionary

## Node Features (`node_features.parquet`)
- **`ticker`**: Stock symbol.
- **`date`**: Observation date.
- **`feature_0` to `feature_31`**: 32-dimensional normalized feature vector including technical and fundamental indicators.

## Edges
- **`correlation_edges.parquet`**: Edges between stocks with historically high price correlation. Includes `edge_attr` indicating the correlation coefficient.
- **`sentiment_edges.parquet`**: Edges representing co-mentions in financial news (via GDELT).
- **`supply_chain_edges.parquet`**: Edges between supplier and consumer companies.
- **`sector_edges.parquet`**: Fully connected components for companies in the same GICS sector.
- **`fundamental_edges.parquet`**: Edges linking companies with similar financial ratios.
