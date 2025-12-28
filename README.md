Graph Data Structure for Spatial Queries  
Overview  

This project builds a KD-Tree (K-Dimensional Tree) from scratch in Python to handle spatial queries on 2D point data effectively. The main goal is to show how spatial indexing can greatly improve the performance of nearest-neighbor and range queries compared to simple brute-force methods.  

The implementation emphasizes clarity, correctness, and performance evaluation using standard Python tools, making it suitable for academic submission and learning.  

Problem Statement  

Spatial datasets often contain large collections of points, such as geographic coordinates, feature points in images, or object locations in simulations. Using brute-force methods for proximity queries becomes inefficient as the dataset grows.  

This project addresses the problem by implementing a KD-Tree to:  

1. Organize 2D spatial data efficiently  
2. Reduce unnecessary distance calculations during queries  
3. Enable faster search operations through spatial pruning  

Objectives  

The main objectives of this project are:  

1. To implement a KD-Tree data structure from scratch for indexing 2D points  
2. To support k-Nearest Neighbor (k-NN) search  
3. To support radius-based (range) search  
4. To compare the performance of KD-Tree queries with brute-force search  
5. To measure query execution time using Python’s timeit module  

Technologies and Tools Used  

- Python 3 – Core programming language  
- NumPy – Efficient numerical and vector operations  
- timeit – Accurate performance benchmarking  

No external KD-Tree or spatial libraries are used. The entire data structure and search logic are implemented manually.  

KD-Tree Approach  

A KD-Tree is a binary tree that organizes points in a k-dimensional space. In this project:  

- The data is two-dimensional (2D).  
- The tree splits points alternately along the x-axis and y-axis.  
- Each node represents a single point in space.  
- Left and right subtrees represent spatial divisions.  

This recursive space-partitioning allows search algorithms to prune entire regions that cannot have relevant results, which improves efficiency.  

Supported Spatial Queries  
1. k-Nearest Neighbor (k-NN) Search  

The k-NN query finds the k closest points to a given query point. The KD-Tree traversal prioritizes branches closer to the query point and prunes branches that cannot contain closer neighbors.  

2. Radius (Range) Search  

The radius search retrieves all points within a specified distance from a query point. The KD-Tree avoids exploring subtrees whose spatial regions lie completely outside the search radius.  

Brute-Force Comparison  

To evaluate the KD-Tree's efficiency:  

- A brute-force approach calculates distances between the query point and all points in the dataset.  
- Both KD-Tree and brute-force methods are tested using the same queries.  
- Execution times are measured with the timeit module.  

This comparison highlights the advantage of spatial indexing as the dataset size increases.  

Benchmarking and Performance Evaluation  

Benchmarking includes:  

- Running multiple k-NN and radius queries  
- Measuring average execution time for KD-Tree searches  
- Measuring average execution time for brute-force searches  

The results show that:  

- Brute-force search time increases linearly with the number of points.  
- KD-Tree search time increases much more slowly due to effective pruning.  

Testing and Validation  

Basic unit tests ensure:  

1. Correct KD-Tree construction  
2. Valid k-NN results  
3. Accurate radius search outputs  

These tests verify correctness before performance evaluation.  

Deliverables Included  

This project meets all required deliverables:  

- A Python module containing the KDTree class  
- Support for k-NN and radius search queries  
- Brute-force implementations for comparison  
- Benchmark scripts and timing results  
- This README explaining usage, approach, and results  

Conclusion  

This project shows how KD-Trees provide an efficient solution for spatial queries in two-dimensional data. By organizing data through spatial partitioning, the KD-Tree greatly reduces the number of distance calculations needed during search operations.  

The comparison with brute-force methods clearly demonstrates the benefits of spatial data structures for scalable and performance-critical applications.  

Applications  

The concepts in this project apply to real-world areas such as:  

- Geographic Information Systems (GIS)  
- Computer Vision and image processing  
- Machine learning (nearest-neighbor search)  
- Robotics and navigation  
- Spatial databases  
