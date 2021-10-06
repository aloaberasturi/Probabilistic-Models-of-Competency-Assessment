import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

"""
Creates the assessment and homophily networks
"""

class Graph:
    """
    Custom graph class. Creates a graph with S nodes. Each node is assessed n times
    """
    def __init__(self, S, n):
        self.S = S
        self.n = n
        self.edges = self._compute_edges()
        self.nodes = np.arange(0, S)

    def _compute_edges(self):
        combinations = list(itertools.permutations(list(range(1,self.S+1)),2)) # max num combinations = S^{2}-S
        flatten = lambda t: [item for sublist in t for item in sublist]    
        uv = flatten([random.sample([i for i in combinations if i[0] == k],self.n) for k in range(1, self.S+1)])
        return [(v,u) for u,v in uv]
        # first element referes to gradee, second element refers to grader


def set_assessment_graph(S, n):
    # my method to generate edges:
    # 
    return nx.random_regular_graph(n, S).to_directed()    


def set_homophily_graph(S, m):
    return nx.barabasi_albert_graph(S, m)    
    # nx.draw(G, with_labels=True)
    # plt.show()







