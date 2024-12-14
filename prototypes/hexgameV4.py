"""
Hex game winner prediction by using connectivity and empty cells.

Works with build of GraphTsetlinMachine from commit ed16ef4b574549fa3bb15110dc0cdcb41de8225d of https://github.com/cair/GraphTsetlinMachine
"""
from utils import load_dataset, booleanize_positions, create_graph
from networkx import has_path
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import numpy as np
import argparse
from tqdm import tqdm
from skimage.util import view_as_windows
from time import time
from pathlib import Path

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=2048, type=int)
    parser.add_argument("--hypervector-bits", default=4, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=5000, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

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

    TODO: Speeding it up 
    - Creating the networkx graph may be slow, maybe consider inlining a pathfinding algorithm directly on the position?
    """
    graph = create_graph(position)
    conmat = np.zeros((2, 49, 49), dtype=np.uint8)
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
                conmat[p, i, j] = 1 if has_path(graph, (x1, y1), (x2, y2)) else 0
    
    return conmat


def save_prepared_dataset(X, Y):
    assert X.shape[0] == Y.shape[0]
    with open(Path(__file__).parent / "hexgamev4-prepared-dataset.csv", mode="w+") as csv:
        for i in tqdm(range(X.shape[0]), desc = "Saving prepared dataset to file"):
            line = ""
            for x in X[i]:
                line += str(x)+","
            line += str(Y[i])
            csv.write(line + "\n")

def load_prepared_dataset(num_rows, file_name):
    X = []
    Y = []
    with open(Path(__file__).parent / "hexgamev4-prepared-dataset.csv") as csv:
        for line_number, line in enumerate(tqdm(csv, desc = "Loading prepared dataset", unit = "Rows", total = num_rows)):
            line = line.split(",")
            x = []
            for entry in line[:-1]:
                x.append(int(entry))
            Y.append(int(line[-1]))
            X.append(x)
            if line_number + 1 == num_rows: break

    return np.array(X), np.array(Y)

args = default_args()

def create_dataset(num_rows, save):
    positions, Y = load_dataset("hex_games_1_000_000_size_7.csv", num_rows)

    Y = np.where(Y > 0, 1, 0)
    X = []

    for i in tqdm(range(positions.shape[0]), desc = "Extracting features"):
        connectivity_features = create_connectivity_matrix(positions[i])
        empty_slots_features = np.where([i] == 0, 1, 0)
        x = []
        for j in connectivity_features.reshape(-1):
            x.append(j)
        for j in empty_slots_features.reshape(-1):
            x.append(j)
        X.append(x)
    
    X = np.array(X)

    if save:
        save_prepared_dataset(X, Y)

    return X, Y

num_rows = 1000
#X, Y = create_dataset(num_rows = num_rows, save = True)
X,Y = load_prepared_dataset(num_rows = num_rows, file_name = "hexgamev4-prepared-dataset.csv") 

# First 80% of data is training, the remaining is test
split_index = int(0.8 * num_rows)
X_train = X[:split_index]
Y_train = Y[:split_index]
X_test = X[split_index:]
Y_test = Y[split_index:]

train_counts = np.array(np.unique(Y_train, return_counts=True)).T
test_counts = np.array(np.unique(Y_test, return_counts=True)).T

print("Train balance:")
print(train_counts)

print("Test balance:")
print(test_counts)

symbols = []
for i in range(49 * 49 * 2 + 49):
    symbols.append("x:%d" % (i))

graphs_train = Graphs(
    X_train.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)

# Initialize graphs

# Train graphs
number_of_nodes = 1
for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)
    
graphs_train.prepare_node_configuration()

# Add nodes
for graph_id in range(X_train.shape[0]):
    graphs_train.add_graph_node(graph_id, "con", 0)
        
graphs_train.prepare_edge_configuration()

for graph_id in tqdm(range(X_train.shape[0]), desc = "Producing training data"):
    for k in X_train[graph_id].nonzero()[0]:
        graphs_train.add_graph_node_property(graph_id, "con", "x:%d" % (k))

graphs_train.encode()


# Test graphs
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    graphs_test.add_graph_node(graph_id, "con", 0)

graphs_test.prepare_edge_configuration()

for graph_id in tqdm(range(X_test.shape[0]), desc = "Producing test data"):
    for k in X_test[graph_id].nonzero()[0]:
        graphs_test.add_graph_node_property(graph_id, "con", "x:%d" % (k))

graphs_test.encode()


tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals
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
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)