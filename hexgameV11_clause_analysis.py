from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import time
import argparse
from utils import load_dataset, create_graph, get_neighbour_lookup, get_all_board_coordinates, get_all_possible_connections, append_to_statistics_file
from networkx import has_path
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split

RED = -1
EMPTY = 0
BLUE = 1

lookups = {}

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--number-of-clauses", default=1_000, type=int)
    parser.add_argument("--T", default=20000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=1, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=1, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=None, type=int)
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
        symbols.append(get_red_symbol(y0, x0))
        symbols.append(get_blue_symbol(y0, x0))
        
    for (y0, x0, y1, x1) in get_all_possible_connections(board_size):
        symbols.append(get_connection_symbol(RED, y0, x0, y1, x1))
        symbols.append(get_connection_symbol(BLUE, y0, x0, y1, x1))
    return symbols
    
def get_connection_symbol(player_color, y0, x0, y1, x1):
    if player_color == RED:
        return f"CR_{x0}_{y0}_{x1}_{y1}"
    elif player_color == BLUE:
        return f"CB_{x0}_{y0}_{x1}_{y1}"
    else:
        raise ValueError(f"Invalid player_color: {player_color}")
    
def get_red_symbol(y, x):
    return f"R_{x}_{y}"
    
def get_blue_symbol(y, x):
    return f"B_{x}_{y}"
    
def populate_graphs(X: np.ndarray, graphs: Graphs, board_size):
    board_coordinates = get_all_board_coordinates(board_size)
    all_possible_connections = get_all_possible_connections(board_size)
    
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
        
        for (x, y) in board_coordinates:
            if board[y,x] == RED:
                graphs.add_graph_node_property(graph_id, node_name, get_red_symbol(y, x))
            elif board[y,x] == BLUE:
                graphs.add_graph_node_property(graph_id, node_name, get_blue_symbol(y, x))
                
        for (y0, x0, y1, x1) in all_possible_connections:
            if y0 == y1 and x0 == x1:
                raise ValueError("This should never happen")
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

def write_output_for_interpretability_analysis(tm: MultiClassGraphTsetlinMachine, args, graphs_train: Graphs, graphs_test: Graphs, Y_test):
    file = open("output.txt", "w+")

    def fprint(line, end = "\n"):
        file.write(f"{line}{end}")

    print("Getting weights")
    weights = tm.get_state()[1].reshape(2, -1)

    #for i in range(tm.number_of_clauses):
    #        fprint("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
    #        l = []
    #        for k in range(args.hypervector_size * 2):
    #            if tm.ta_action(0, i, k):
    #                if k < args.hypervector_size:
    #                    l.append("x%d" % (k))
    #                else:
    #                    l.append("NOT x%d" % (k - args.hypervector_size))

            # for k in range(args.message_size * 2):
            #     if tm.ta_action(1, i, k):
            #         if k < args.message_size:
            #             l.append("c%d" % (k))
            #         else:
            #             l.append("NOT c%d" % (k - args.message_size))

    #        fprint(" AND ".join(l))


    def get_symbol_name_from_symbol_id(id):
        for (k, v) in graphs_test.symbol_id.items():
            if v == id:
                return k
        raise ValueError(f"Could not find symbol name for symbol {id}")
    
    
    print("Getting clauses")
    clauses = tm.get_clause_literals(graphs_test.hypervectors)
    fprint("Symbols:")
    fprint(graphs_test.symbol_id)
    fprint(clauses.shape)
    fprint(clauses[0])
    fprint(clauses)
    num_symbols = len(graphs_test.symbol_id)
    fprint("Clauses:")
    shortest_clause = ""
    shortest_clause_length = float('inf')
    for c in tqdm(range(clauses.shape[0]), desc = "Looping through clauses"):

        # Ignore clauses that are negative for Red player (y = 0)
        if weights[0, c] < 0:
            continue
        
        clause_literals = []
        for i in range(num_symbols * 2):
            if i < num_symbols:
                if clauses[c][i] == 1:
                    clause_literals.append(get_symbol_name_from_symbol_id(i))    
            else:
                if clauses[c][i] == 1:    
                    clause_literals.append(f"NOT {get_symbol_name_from_symbol_id(i-num_symbols)}")
        
        if len(clause_literals) > 0 and len(clause_literals) < shortest_clause_length:
            shortest_clause_length = len(clause_literals)
            shortest_clause = f"Clause {c}: " +  " AND ".join(clause_literals)
            fprint(shortest_clause)
        
    fprint(f"Shortest clause (length {shortest_clause_length}): ")
    fprint(shortest_clause)

    # Symbols in a graph gets encoded as a hypervector
    fprint(graphs_test.hypervectors.shape)
    fprint(bin(graphs_test.hypervectors[2][0]))

    # Each clause is represented by a hypervector?
    fprint(tm.hypervectors.shape)
    fprint(bin(tm.hypervectors[99][0]))
    fprint(graphs_test.edge_type_id)
    
    clause_outputs, class_sums = tm.transform_nodewise(graphs_test)
    # Lets consider example n in graphs_test:
    n = 0
    fprint(f"*** Example {n} ***")
    fprint(f"Y: {Y_test[n]}")
    fprint(f"Class sum: {class_sums[n]}")

    file.close()
    
def train():
    args = default_args()
    # Create train data

    num_rows = 10_000
    board_size = 7
    print("Total number of symbols: ", len(get_all_symbols(board_size)))
    print("Possible connections: ", len(get_all_possible_connections(board_size)))

    #X_train, Y_train = load_dataset("hex_9x9_2moves_train.csv", num_rows = num_rows)
    #X_test, Y_test = load_dataset("hex_9x9_2moves_test.csv", num_rows = num_rows)
    
    X, Y = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    # make -1 correspond to class 0, and 1 to 1
    Y_train = np.where(Y_train > 0, 1, 0)
    Y_test = np.where(Y_test > 0, 1, 0)

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


    number_of_clauses = [1_000]

    TS = time.strftime("%Y%m%d_%H%M%S")
    stats_file_name = f"number_of_clauses_accuracy_{TS}.csv"
    print("Appending statistics to", stats_file_name)
    append_to_statistics_file(stats_file_name, "max accuracy", "number of clauses")

    for nc in number_of_clauses:
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses = nc,
            T = args.T,
            s = 5,
            number_of_state_bits = args.number_of_state_bits,
            depth = args.depth,
            message_size = args.message_size,
            message_bits = args.message_bits,
            max_included_literals = args.max_included_literals,
            double_hashing = args.double_hashing,
            grid=(16*13,1,1),
            block=(128,1,1)
        )
        
        max_accuracy = 0
        for i in range(args.epochs):
            start_training = time.time()
            tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
            stop_training = time.time()
            start_testing = time.time()
            result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
            stop_testing = time.time()
            if result_test > max_accuracy:
                max_accuracy = result_test
                print("New record accuracy:", max_accuracy)
            result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
            print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
            
        append_to_statistics_file(stats_file_name, str(max_accuracy), str(nc))
        
    write_output_for_interpretability_analysis(tm, args, graphs_train, graphs_test, Y_test)


if __name__ == '__main__':  
    train()