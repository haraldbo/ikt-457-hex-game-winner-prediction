"""
Hex game winner prediction using convolutional graph tsetlin machine. Based on example MNISTConvolutionDemo.py

Works with build of GraphTsetlinMachine from commit ed16ef4b574549fa3bb15110dc0cdcb41de8225d of https://github.com/cair/GraphTsetlinMachine
"""
from utils import load_dataset, booleanize_positions
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import numpy as np
import argparse
from tqdm import tqdm
from skimage.util import view_as_windows
from time import time

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

num_rows = 20000
positions, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)

winners = np.where(winners > 0, 1, 0) 
positions = booleanize_positions(positions)

# First 80% of data is training, the remaining is test
split_index = int(0.8 * num_rows)
train_positions = positions[:split_index]
train_winners = winners[:split_index]
test_positions = positions[split_index:]
test_winners = winners[split_index:]

patch_size = 5

row_dims = 14 - patch_size + 1
col_dims = 7 - patch_size + 1

symbols = []
for i in range(row_dims):
    symbols.append("R:%d" % (i))

for i in range(col_dims):
    symbols.append("C:%d" % (i))
    
for i in range(patch_size * patch_size):
    symbols.append(i)


#print("Symbols:")
#print(symbols)

graphs_train = Graphs(
    train_positions.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)

# Initialize graphs
number_of_nodes = row_dims * col_dims
for graph_id in range(train_positions.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)
    
graphs_train.prepare_node_configuration()

# Add nodes
for graph_id in range(train_positions.shape[0]):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        graphs_train.add_graph_node(graph_id, node_id, 0)
        
graphs_train.prepare_edge_configuration()

for graph_id in tqdm(range(train_positions.shape[0]), desc = "Producing training data"):
     
    windows = view_as_windows(train_positions[graph_id,:,:], (patch_size, patch_size))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                #node_id = q*dim + r
                node_id = q*col_dims + r

                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_train.add_graph_node_property(graph_id, node_id, k)

                graphs_train.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
                graphs_train.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

graphs_train.encode()

graphs_test = Graphs(test_positions.shape[0], init_with=graphs_train)
for graph_id in range(test_positions.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(test_positions.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        graphs_test.add_graph_node(graph_id, node_id, 0)

graphs_test.prepare_edge_configuration()

for graph_id in tqdm(range(test_positions.shape[0]), desc = "Producing test data"):
     
    windows = view_as_windows(test_positions[graph_id,:,:], (patch_size, patch_size))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q*col_dims + r

                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_test.add_graph_node_property(graph_id, node_id, k)

                graphs_test.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
                graphs_test.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

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
    tm.fit(graphs_train, train_winners, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == test_winners).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == train_winners).mean()

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
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, train_winners, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_test) == test_winners).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == train_winners).mean()

print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)