from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import random
from utils import load_dataset, create_graph, display_as_graph
import networkx

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--number-of-clauses", default=5000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=256, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=2000, type=int)
    parser.add_argument("--number-of-state-bits", default=8, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def get_node_name(y, x):
    return f"y{y}x{x}"

def get_color(p):
    if p == -1:
        return "B"
    if p == 0:
        return "E"
    if p == 1:
        return "W"

args = default_args()

# Create train data

num_rows = 1000
board_size = 7
X, Y = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)

Y = np.where(Y > 0, 1, 0)

# First 80% of data is training, the remaining is test
split_index = int(0.8 * num_rows)
X_train = X[:split_index]
Y_train = Y[:split_index]
X_test = X[split_index:]
Y_test = Y[split_index:]

graphs_train = Graphs(
    X_train.shape[0],
    symbols=["B", "E", "W"],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, board_size * board_size)

graphs_train.prepare_node_configuration()

for graph_id in range(X_train.shape[0]):
    board = X_train[graph_id]
    g = create_graph(board)
    for (y, x) in g.nodes:
        num_edges = len(g.edges((y, x)))
        graphs_train.add_graph_node(graph_id, get_node_name(y, x), num_edges)

graphs_train.prepare_edge_configuration()

for graph_id in range(X_train.shape[0]):
    edge_type = "Plain"
    board = X_train[graph_id]
    g = create_graph(board)
    for (src_node, dest_node) in g.edges:
        graphs_train.add_graph_node_edge(graph_id, get_node_name(*src_node), get_node_name(*dest_node), edge_type)
        graphs_train.add_graph_node_edge(graph_id, get_node_name(*dest_node), get_node_name(*src_node), edge_type)

for graph_id in range(X_train.shape[0]):
    board = X_train[graph_id]
    for y in range(board_size):
        for x in range(board_size):
            graphs_train.add_graph_node_property(graph_id, get_node_name(y, x), get_color(board[y, x]))

graphs_train.encode()


print("Creating testing data")

graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, board_size * board_size)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    board = X_test[graph_id]
    g = create_graph(board)
    for (y, x) in g.nodes:
        num_edges = len(g.edges((y, x)))
        graphs_test.add_graph_node(graph_id, get_node_name(y, x), num_edges)

graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    edge_type = "Plain"
    board = X_test[graph_id]
    g = create_graph(board)
    for (src_node, dest_node) in g.edges:
        graphs_test.add_graph_node_edge(graph_id, get_node_name(*src_node), get_node_name(*dest_node), edge_type)
        graphs_test.add_graph_node_edge(graph_id, get_node_name(*dest_node), get_node_name(*src_node), edge_type)

for graph_id in range(X_test.shape[0]):
    board = X_test[graph_id]
    for y in range(board_size):
        for x in range(board_size):
            graphs_test.add_graph_node_property(graph_id, get_node_name(y, x), get_color(board[y, x]))

graphs_test.encode()

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

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

exit(0)

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