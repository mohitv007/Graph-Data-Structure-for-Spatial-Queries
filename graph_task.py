"""
Graph Data Structure for Spatial Queries
KD-Tree implementation for 2D points with k-NN and Radius Search
Includes brute-force comparison and benchmarking
"""

import numpy as np
import heapq
import timeit


# =========================
# KD-TREE IMPLEMENTATION
# =========================

class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, points):
        self.k = 2  # 2D space
        self.root = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        if len(points) == 0:
            return None

        axis = depth % self.k
        points = points[points[:, axis].argsort()]
        median = len(points) // 2

        return KDNode(
            point=points[median],
            left=self._build_tree(points[:median], depth + 1),
            right=self._build_tree(points[median + 1:], depth + 1)
        )

    def _distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # =========================
    # k-NN SEARCH (FIXED)
    # =========================
    def knn_search(self, query_point, k):
        heap = []
        counter = 0  # tie-breaker to avoid numpy comparison

        def search(node, depth):
            nonlocal counter
            if node is None:
                return

            axis = depth % self.k
            dist = self._distance(query_point, node.point)

            entry = (-dist, counter, node.point)
            counter += 1

            if len(heap) < k:
                heapq.heappush(heap, entry)
            else:
                if dist < -heap[0][0]:
                    heapq.heappushpop(heap, entry)

            diff = query_point[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

            search(near, depth + 1)

            if len(heap) < k or abs(diff) < -heap[0][0]:
                search(far, depth + 1)

        search(self.root, 0)
        return [p for (_, _, p) in heap]

    # =========================
    # RADIUS SEARCH
    # =========================
    def radius_search(self, query_point, radius):
        result = []

        def search(node, depth):
            if node is None:
                return

            axis = depth % self.k
            dist = self._distance(query_point, node.point)

            if dist <= radius:
                result.append(node.point)

            diff = query_point[axis] - node.point[axis]

            if diff < 0:
                search(node.left, depth + 1)
                if abs(diff) <= radius:
                    search(node.right, depth + 1)
            else:
                search(node.right, depth + 1)
                if abs(diff) <= radius:
                    search(node.left, depth + 1)

        search(self.root, 0)
        return result


# =========================
# BRUTE FORCE METHODS
# =========================

def brute_knn(points, query, k):
    distances = [(np.linalg.norm(p - query), p) for p in points]
    distances.sort(key=lambda x: x[0])
    return [p for _, p in distances[:k]]


def brute_radius(points, query, radius):
    return [p for p in points if np.linalg.norm(p - query) <= radius]


# =========================
# UNIT TESTS
# =========================

def run_tests():
    print("Running basic tests...")

    points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    tree = KDTree(points)
    query = np.array([2, 3])

    knn = tree.knn_search(query, k=2)
    assert len(knn) == 2

    radius_result = tree.radius_search(query, radius=3)
    assert len(radius_result) > 0

    print("All tests passed ✔")


# =========================
# BENCHMARKING
# =========================

def benchmark():
    print("\nBenchmarking KD-Tree vs Brute Force")

    points = np.random.rand(10000, 2)
    query = np.array([0.5, 0.5])
    k = 5
    radius = 0.1

    tree = KDTree(points)

    kd_knn_time = timeit.timeit(
        lambda: tree.knn_search(query, k),
        number=100
    )

    bf_knn_time = timeit.timeit(
        lambda: brute_knn(points, query, k),
        number=100
    )

    kd_radius_time = timeit.timeit(
        lambda: tree.radius_search(query, radius),
        number=100
    )

    bf_radius_time = timeit.timeit(
        lambda: brute_radius(points, query, radius),
        number=100
    )

    print(f"KD-Tree kNN Time     : {kd_knn_time:.6f} s")
    print(f"Brute-force kNN Time : {bf_knn_time:.6f} s")
    print(f"KD-Tree Radius Time  : {kd_radius_time:.6f} s")
    print(f"Brute-force Radius  : {bf_radius_time:.6f} s")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run_tests()
    benchmark()
