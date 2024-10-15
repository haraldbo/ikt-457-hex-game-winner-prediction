#!/usr/bin/python

import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import MultiClassTsetlinMachine
from utils import load_dataset, create_graph
from networkx import has_path
import numpy as np
from tqdm import tqdm
from utils import load_dataset
import numpy as np

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


num_rows = 50_000
positions, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)

Y = np.where(winners > 0, 1, 0)
X = []

# TODO: This is very slow, maybe do it for the whole dataset and save it to a file
# Then we can load preprocessed data each time
for position in tqdm(positions, desc = "Creating connectivity matrices"):
  conmat = create_connectivity_matrix(position)
  empty_slots = np.where(position == 0, 1, 0)
  x = []
  for i in conmat.flatten():
      x.append(i)
  for i in empty_slots.flatten():
      x.append(i)
  X.append(x)
X = np.array(X)

split_index = int(0.8 * num_rows)
X_train = X[:split_index]
Y_train = Y[:split_index]

X_test = X[split_index:]
Y_test = Y[split_index:]

# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = 2000
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 49 * 49 * 2 + 49
number_of_classes = 2

# Training configuration
epochs = 1

print("Init TM")
tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

print("Starting training")
#tsetlin_machine.fit(X_train, Y_train, Y_train.shape[0], epochs=epochs)
print("Done training")

batch_size = 50
current_batch = 50

# TODO implement batch training and testing:
for i in range(1000):
  # Maybe use np.arrange?
  X_batch = X_train[current_batch-batch_size:current_batch]
  Y_batch = Y_train[current_batch-batch_size:current_batch]

  tsetlin_machine.fit(X_batch, Y_batch, Y_batch.shape[0], epochs=epochs)
  print("Accuracy on batch data:", tsetlin_machine.evaluate(X_batch, Y_batch, Y_batch.shape[0]))
  current_batch += batch_size

#print("Accuracy on test data:", tsetlin_machine.evaluate(X_test, Y_test, Y_test.shape[0]))
#print("Accuracy on training data:", tsetlin_machine.evaluate(X_train, Y_train, Y_train.shape[0]))

