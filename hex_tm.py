from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import time
from utils import load_dataset, create_graph, get_all_board_coordinates, get_all_possible_connections, append_to_statistics_file
from networkx import has_path
from tqdm import tqdm

RED = -1
EMPTY = 0
BLUE = 1

def get_all_symbols(board_size):
    symbols = []
    board_coordinates = get_all_board_coordinates(board_size)
    for (y0, x0) in board_coordinates:
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

def write_output_for_interpretability_analysis(tm: MultiClassGraphTsetlinMachine, graphs_train: Graphs, graphs_test: Graphs, Y_test):
    file = open("clause_analysis.txt", "w+")

    def fprint(line, end = "\n"):
        file.write(f"{line}{end}")

    print("Getting weights")
    weights = tm.get_state()[1].reshape(2, -1)
    print(weights.shape)


    def get_symbol_name_from_symbol_id(id):
        for (k, v) in graphs_train.symbol_id.items():
            if v == id:
                return k
        raise ValueError(f"Could not find symbol name for symbol {id}")
    
    
    print("Getting clauses")
    clauses = tm.get_clause_literals(graphs_train.hypervectors)
    fprint(clauses.shape)
    fprint(clauses[0])
    fprint(clauses)
    num_symbols = len(graphs_train.symbol_id)
    fprint("*** Clauses ***")
    shortest_clause = ""
    shortest_clause_length = float('inf')
    for c in tqdm(range(clauses.shape[0]), desc = "Looping through clauses"):

        # Ignore clauses that are negative for player
        
        if weights[1, c] < 0: # Blue
        #if weights[0, c] < 0: # Red
            continue
        
        clause_literals = []
        for i in range(num_symbols * 2):
            if i < num_symbols:
                if clauses[c][i] == 1:
                    clause_literals.append(get_symbol_name_from_symbol_id(i))    
            else:
                if clauses[c][i] == 1:    
                    clause_literals.append(f"NOT {get_symbol_name_from_symbol_id(i-num_symbols)}")

        clause = f"Clause {c} (length {len(clause_literals)}): " +  " AND ".join(clause_literals)
        fprint(clause)
        if len(clause_literals) > 0 and len(clause_literals) < shortest_clause_length:
            shortest_clause_length = len(clause_literals)
            shortest_clause = clause
        
    fprint(f"Shortest clause (length {shortest_clause_length}): ")
    fprint(shortest_clause)

    file.close()
    
def train():
    board_size = 9
    epochs = 10
    number_of_clauses = [5000]
    s_values = [12]
    hv_size = 512
    hv_bits = 2
    T_value = 20_000
    
    print("Total number of symbols: ", len(get_all_symbols(board_size)))
    print("Possible connections: ", len(get_all_possible_connections(board_size)))

    X_train, Y_train = load_dataset("hex_9x9_2moves_train.csv")
    X_test, Y_test = load_dataset("hex_9x9_2moves_test.csv")
    
    #X, Y = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = num_rows)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
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
        hypervector_size=hv_size,
        hypervector_bits=hv_bits,
    )
    graphs_train = populate_graphs(X_train, graphs_train, board_size)
    print("Done.")

    print("Creating test graphs.")
    graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
    graphs_test = populate_graphs(X_test, graphs_test, board_size)
    print("Done.")

    TS = time.strftime("%Y%m%d_%H%M%S")
    stats_file_name = f"number_of_clauses_accuracy_{TS}.csv"
    print("Storing training statistics in ", stats_file_name)
    append_to_statistics_file(stats_file_name, "max accuracy", "number of clauses", "s")

    for nc in number_of_clauses:
        for s in s_values:
            tm = MultiClassGraphTsetlinMachine(
                number_of_clauses = nc,
                T = T_value,
                s = s,
                number_of_state_bits = 16,
                depth = 1,
                message_size = 256,
                message_bits = 1,
                max_included_literals = None,
                double_hashing = False,
                grid=(16*13,1,1),
                block=(128,1,1)
            )

            max_accuracy = 0
            for i in range(epochs):
                start_training = time.time()
                tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
                stop_training = time.time()
                start_testing = time.time()
                result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
                stop_testing = time.time()
                if result_test > max_accuracy:
                    max_accuracy = result_test
                    print("New accuracy record:", max_accuracy)
                result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
                print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
            
            append_to_statistics_file(stats_file_name, str(max_accuracy), str(nc), str(s))
        
    write_output_for_interpretability_analysis(tm, graphs_train, graphs_test, Y_test)

if __name__ == '__main__':  
    train()