import networkx as nx
from networkx import has_path
from random import randint
import numpy as np
from utils import display_as_graph

def get_winner(board_graph, board_size):
    """
    Returns player that has won. Returns 0 if no winner.
    
    -1 needs to make a connection from first row to last row
    1 needs to make a connection from first column to last column

    """
    # Check if 1 has won
    for i in range(board_size):
        fy = i
        fx = 0
        if board_graph.nodes[(fy, fx)]["piece"] != 1:  continue
        for j in range(board_size):
            ty = j
            tx = board_size - 1
            if board_graph.nodes[(ty, tx)]["piece"] != 1: continue
            if has_path(board_graph, (fy, fx), (ty, tx)): return 1

    # Check if -1 has won
    for i in range(board_size):
        fy = 0
        fx = i
        if board_graph.nodes[(fy, fx)]["piece"] != -1: continue
        for j in range(board_size):
            ty = board_size - 1
            tx = j
            if board_graph.nodes[(ty, tx)]["piece"] != -1: continue
            if has_path(board_graph, (fy, fx), (ty, tx)): return -1
        

    return 0

def add_piece(board_graph: nx.Graph, board_size, y, x, piece):
    board_graph.add_node((y, x), piece=piece)
    # Add edge to nearby nodes
    neighbours = []
    if x < board_size-1: # Right neighbour
        neighbours.append((y, x+1))
    if x > 0: # Left neighbour
        neighbours.append((y, x-1))
    if y > 0: # Neighbours above
        neighbours.append((y-1, x))
        if x < board_size-1:
            neighbours.append((y-1, x+1))
    if y < board_size-1: # Neighbours below
        neighbours.append((y+1, x))
        if x > 0:
            neighbours.append((y+1, x-1))

    for (ney, nex) in neighbours:
        neighbour_piece = board_graph.nodes[(ney, nex)]["piece"]
        if piece == neighbour_piece:
            board_graph.add_edge((y, x), (ney, nex))

def create_empty_board_graph(board_size):
    board_graph = nx.Graph()
    for y in range(board_size):
        for x in range(board_size):
            board_graph.add_node((y, x), piece = 0)
    return board_graph

def board_graph_as_matrix(board_graph: nx.Graph, board_size):
    mat = np.zeros((board_size, board_size))
    for y in range(board_size):
        for x in range(board_size):
            mat[y,x] = board_graph.nodes[(y, x)]["piece"]
    return mat

def create_random_game(board_size):
    possible_moves = [(y, x) for y in range(board_size) for x in range(board_size)]
    board_graph = create_empty_board_graph(board_size)
    current_player = 1
    winner = 0
    while winner == 0:
        (y, x) = possible_moves.pop(randint(0, len(possible_moves)-1))
        add_piece(board_graph, board_size, y, x, current_player)
        current_player *= -1 # 1 becomes -1, -1 becomes 1, alternating
        winner = get_winner(board_graph, board_size)
    
    return board_graph_as_matrix(board_graph, board_size), winner

board, winner = create_random_game(9)