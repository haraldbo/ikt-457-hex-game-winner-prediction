"""
Does not work
"""

from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import random
from utils import load_dataset, create_graph, display_as_graph, get_neighbour_lookup
import networkx
from tqdm import tqdm

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

def get_node_name(y, x):
    return f"y{y}x{x}"

def get_node_symbol(p):
    if p == -1:
        return "B"
    if p == 0:
        return "E"
    if p == 1:
        return "W"
    
def populate_graphs(X: np.ndarray, graphs: Graphs, board_size):
    neighbour_lookup = get_neighbour_lookup(board_size)
    
    progress_bar = tqdm(total = X.shape[0] * 3, desc = "Creating graphs", leave = False)
    
    progress_bar.set_description("Setting number of nodes")
    for graph_id in range(X.shape[0]):
        progress_bar.update(1)
        graphs.set_number_of_graph_nodes(graph_id, board_size * board_size)
    
    graphs.prepare_node_configuration()
    
    progress_bar.refresh()
    progress_bar.set_description("Adding nodes")
    for graph_id in range(X.shape[0]):
        progress_bar.update(1)
        board = X[graph_id]
        for y in range(board_size):
            for x in range(board_size):
                num_edges = 0
                for (ney, nex) in neighbour_lookup[y,x]:
                    if board[y,x] == board[ney,nex] and board[y,x] != 0:
                        num_edges += 1
                graphs.add_graph_node(graph_id, get_node_name(y, x), num_edges)
    
    graphs.prepare_edge_configuration()
    
    progress_bar.refresh()
    progress_bar.set_description("Adding edges and node properties")
    for graph_id in range(X.shape[0]):
        progress_bar.update(1)
        edge_type = "Plain"
        board = X[graph_id]
        for y in range(board_size):
            for x in range(board_size):
                graphs.add_graph_node_property(graph_id, get_node_name(y, x), get_node_symbol(board[y, x]))
                
                # Add edge between neighbouring nodes if they are of same color
                for (ney, nex) in neighbour_lookup[(y,x)]:
                    if board[y,x] == board[ney,nex] and board[y,x] != 0:
                        graphs.add_graph_node_edge(graph_id, get_node_name(y, x), get_node_name(ney, nex), edge_type)
    progress_bar.refresh()
    graphs.encode()
    return graphs
    
args = default_args()

# Create train data

num_rows = 10000
board_size = 7
X, Y = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)

Y = np.where(Y > 0, 1, 0)

# First 80% of data is training, the remaining is test
split_index = int(0.8 * num_rows)
X_train = X[:split_index]
Y_train = Y[:split_index]
X_test = X[split_index:]
Y_test = Y[split_index:]

print("Creating training graphs.")
graphs_train = Graphs(
    X_train.shape[0],
    symbols=["B", "E", "W"],
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