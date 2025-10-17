# K-Means Clustering

Unsupervised learning algorithm that partitions data into k clusters by minimizing within-cluster sum of squares (WCSS).

## Overview

K-Means clustering groups similar data points together and identifies underlying patterns by finding k centroids that minimize the distance between data points and their assigned cluster centers.

## Directory Structure

```
13-K Means Clustering Unsupervised/
├── 1-K Means Clustering Implementation.ipynb  # Practical implementation
├── 2-K-means-theory.ipynb                    # Mathematical concepts
├── WCSS.png                                  # Elbow method visualization
└── README.md                                 # This file
```

## Key Concepts

- **Centroids**: Cluster centers that minimize intra-cluster distances
- **WCSS**: Within-Cluster Sum of Squares - optimization objective
- **Elbow Method**: Technique to find optimal number of clusters
- **Convergence**: Algorithm stops when centroids no longer move significantly

## Algorithm Steps

1. **Initialize** k random centroids
2. **Assign** each point to nearest centroid
3. **Update** centroids to cluster means
4. **Repeat** steps 2-3 until convergence



## Key Parameters

- **n_clusters**: Number of clusters (k)
- **init**: Initialization method ('k-means++', 'random')
- **max_iter**: Maximum iterations (default: 300)
- **tol**: Tolerance for convergence (default: 1e-4)
- **random_state**: Seed for reproducibility

## Use Cases

- **Customer Segmentation**: Group customers by behavior
- **Market Research**: Identify consumer segments  
- **Image Segmentation**: Separate image regions
- **Data Compression**: Reduce data complexity
- **Anomaly Detection**: Identify outliers
- **Gene Analysis**: Group similar gene expressions

## Advantages

✅ **Simple**: Easy to understand and implement
✅ **Fast**: Computationally efficient for large datasets
✅ **Scalable**: Works well with high-dimensional data
✅ **Guaranteed Convergence**: Always converges to local optimum

## Disadvantages

❌ **Choose k**: Need to specify number of clusters
❌ **Sensitive to Initialization**: Different starts may give different results
❌ **Spherical Clusters**: Assumes clusters are spherical and similar sized
❌ **Outliers**: Sensitive to outliers and noise
❌ **Local Optimum**: May not find global optimum


## Best Practices

1. **Scale Features**: Standardize data before clustering
2. **Choose k Wisely**: Use elbow method or silhouette analysis
3. **Multiple Runs**: Run algorithm multiple times with different initializations
4. **Validate Results**: Use domain knowledge to interpret clusters
5. **Handle Outliers**: Consider removing or treating outliers separately


---

*K-Means is the most popular clustering algorithm, perfect for discovering hidden patterns in unlabeled data.*