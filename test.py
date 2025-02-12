from genome import Genome
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque


class Graph:
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.in_degree = [0] * n
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.in_degree[v] += 1

g = Graph(4)
g.add_edge(1,3)
print("hi")