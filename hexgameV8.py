"""
Based on connections and empty slots
"""

from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from utils import load_dataset, create_graph, get_neighbour_lookup, get_all_board_coordinates
from networkx import has_path
from tqdm import tqdm
import networkx as nx

BLACK = -1
EMPTY = 0
WHITE = 1

lookups = {}

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3000, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=20000, type=int)
    parser.add_argument("--s", default=15.0, type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=500, type=int)
    parser.add_argument("--number-of-state-bits", default=16, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


def get_all_symbols(board_size):
    symbols = []
    board_coordinates = get_all_board_coordinates(board_size)
    for (y0, x0) in board_coordinates:
        # Empty slots
        symbols.append(get_empty_symbol(y0, x0))
        for (y1, x1) in board_coordinates:
            # White connection from i to j
            symbols.append(get_connection_symbol(WHITE, y0, x0, y1, x1))

            # Black connection from i to j
            symbols.append(get_connection_symbol(BLACK, y0, x0, y1, x1))
    return symbols
    
def get_connection_symbol(player_color, y0, x0, y1, x1):
    if player_color == BLACK:
        return f"CB_{y0}_{x0}_{y1}_{x1}"
    elif player_color == WHITE:
        return f"CW_{y0}_{x0}_{y1}_{x1}"
    else:
        raise ValueError(f"Invalid player_color: {player_color}")
    
def get_empty_symbol(y, x):
    return f"E_{y}_{x}"
    
def populate_graphs(X: np.ndarray, graphs: Graphs, board_size):
    board_coordinates = get_all_board_coordinates(board_size)
    
    progress_bar = tqdm(total = X.shape[0] * 3, desc = "Creating graphs", leave = False)
    node_name = "The One"
    progress_bar.set_description("Setting number of nodes")
    for graph_id in range(X.shape[0]):
        graphs.set_number_of_graph_nodes(graph_id, 1)
        progress_bar.update(1)
        progress_bar.refresh()
    
    
    graphs.prepare_node_configuration()
    
    progress_bar.set_description("Adding nodes")
    for graph_id in range(X.shape[0]):
        graphs.add_graph_node(graph_id, node_name, 0)
        progress_bar.update(1)
        progress_bar.refresh()
    
    graphs.prepare_edge_configuration()
    
    progress_bar.set_description("Adding node properties")
    for graph_id in range(X.shape[0]):
        
        board = X[graph_id]
        board_graph = create_graph(board)
        
        for (y0, x0) in board_coordinates:
            if board[y0,x0] == EMPTY:
                graphs.add_graph_node_property(graph_id, node_name, get_empty_symbol(y0, x0))
                continue
            
            for (x1, y1) in board_coordinates:
                if board[y0, x0] == board[y1, x1] and has_path(board_graph, (y0, x0), (y1, x1)):
                    graphs.add_graph_node_property(
                        graph_id, 
                        node_name, 
                        get_connection_symbol(board[y0, x0], y0, x0, y1, x1)
                    )
        progress_bar.update(1)
        progress_bar.refresh()
                    
    graphs.encode()
    return graphs
    
args = default_args()

# Create train data

num_rows = None
board_size = 9
X, Y = load_dataset("hex_9x9_2moves.csv", num_rows = num_rows)

Y = np.where(Y > 0, 1, 0)

# First 80% of data is training, the remaining is test
split_index = int(0.8 * len(X))
X_train = X[:split_index]
Y_train = Y[:split_index]
X_test = X[split_index:]
Y_test = Y[split_index:]

print("Train balance:")
unique, counts = np.unique(Y_train, return_counts=True)
print(np.asarray((unique, counts)).T)
print(counts/counts.sum())

print("Test balance:")
unique, counts = np.unique(Y_test, return_counts=True)
print(np.asarray((unique, counts)).T)
print(counts/counts.sum())

print("Creating training graphs.")
graphs_train = Graphs(
    X_train.shape[0],
    symbols=get_all_symbols(board_size),
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
)
graphs_train = populate_graphs(X_train, graphs_train, board_size)
print("Done.")

print("Creating test graphs.")
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
graphs_test = populate_graphs(X_test, graphs_test, board_size)
print("Done.")

print("Initializing tsetlin machine.")
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth = args.depth,
    message_size = args.message_size,
    message_bits = args.message_bits,
    max_included_literals = args.max_included_literals,
    double_hashing = args.double_hashing,
    grid=(16*13,1,1),
    block=(128,1,1)
)

print("Starting training..")
for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))

        # for k in range(args.message_size * 2):
        #     if tm.ta_action(1, i, k):
        #         if k < args.message_size:
        #             l.append("c%d" % (k))
        #         else:
        #             l.append("NOT c%d" % (k - args.message_size))

        print(" AND ".join(l))

print(graphs_test.hypervectors)
print(tm.hypervectors)
print(graphs_test.edge_type_id)