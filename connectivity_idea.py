from utils import load_dataset, create_graph
from networkx import has_path
import numpy as np
from tqdm import tqdm
from utils import load_dataset
import numpy as np
from time import time

def create_connectivity_matrix(position):
    """
    Create and return a 2x49x49 matrix where each entry can be either 0 or 1.
    - 1 when there is a connection between position (x1, y1) and (x2, y2)
    - 0 otherwise

    The first dimension is wheter the player is black/white
    The second dimension is position one (x1, y1) represented as an integer i = x1 + y1 * 7
    The third dimension is position two (x2, y2) represented as an integer j = x2 + y2 * 7

    conmat[0, i, j] contains wheter there is a path between (x1, y1) and (x2, y2), where
    x1 = i % 7
    y1 = i // 7
    x2 = j % 7
    y2 = j // 7

    TODO: Graph can be simplified by
    - All edges are bidirectional

    TODO: 
    - Creating the networkx graph may be slow, maybe consider inlining a pathfinding algorithm directly on the position?
    """
    graph = create_graph(position)
    conmat = np.zeros((2, 49, 49))
    for p in range(2):
        color = [-1, 1][p]
        for i in range(49):
            x1 = i % 7
            y1 = i // 7
            if position[x1, y1] != color: continue
            for j in range(49):
                x2 = j % 7
                y2 = j // 7
                if position[x2, y2] != color: continue
                conmat[p, i, j] = has_path(graph, (x1, y1), (x2, y2))
    
    return conmat


positions, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = 10000)

for position in tqdm(positions, desc = "Creating connectivity matrix"):
    conmat = create_connectivity_matrix(position)
    #print(conmat)


"""
TODO:
1. Load dataset
2. Split
3. For each board, create connectivity matrix
4. For each board, find empty slots
5. Create symbols for connectivity matrix and empty slots
6. Train tsetlin machine to predict winner based on connectivity and empty slots
"""