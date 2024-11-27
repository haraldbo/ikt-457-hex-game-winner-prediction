from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw as draw
import math
from tqdm import tqdm
from random import randint
import networkx as nx


neighbours_lookups = {}
board_coordinates_lookups = {}
all_possible_connections_lookup = {}
    
def load_dataset(file_name, num_rows = None):
    """
    Loads and returns dataset from csv file. 
    ex: "dataset\hex_games_1_000_000_size_7.csv"

    If num_rows is set, then it will load num_rows entries. 
    The first row of the dataset is ignored.
    """
    with open(Path(__file__).parent / "dataset" / file_name) as file:
        file.readline() # Read and ignore line containing headers
        positions = []
        winners = []
        for line_number, line in enumerate(tqdm(file, desc = "Loading dataset", unit = "Rows", total = num_rows)):
            entries = line.split(",")
            board_slots = len(entries)-1
            board_size = int(math.sqrt(board_slots))
            position = [[0 for i in range(board_size)] for j in range(board_size)]
            for i in range(board_slots):
                y = i // board_size
                x = i % board_size
                position[y][x] = entries[i]
            
            winners.append(entries[board_slots])
            positions.append(position)
            
            if (line_number+1) == num_rows:
                break
            
        return np.array(positions, dtype=np.int8), np.array(winners, dtype=np.int8)

def display_position(position: np.ndarray):
    img = np.full((300, 500, 3), 255)

    top = 25
    left = 25
    line_length = 20
    cos30 = math.cos(math.pi/6)
    sin30 = math.sin(math.pi/6)

    board_size = position.shape[0]
    
    def draw_hexagon(x0, y0):    
        lines = [
            # Vertical lines:
            draw.line(y0, x0, y0 + line_length, x0),
            draw.line(y0, int(x0 + 2 * cos30 * line_length), 
                      y0 + line_length, int(x0 + 2 * cos30 * line_length)),

            # Top angled lines
            draw.line(y0, x0, 
                      int(y0 - sin30 * line_length), int(x0 + cos30 * line_length)),
            draw.line(int(y0 - sin30 * line_length), int(x0 + cos30 * line_length), 
                      y0,  int(x0 + 2 * cos30 * line_length)),
            
            # Bottom angled lines
            draw.line(y0 + line_length, x0, 
                      int(y0 + line_length + sin30 * line_length), int(x0 + cos30 * line_length)),
            draw.line(int(y0 + line_length + sin30 * line_length), int(x0 + cos30 * line_length), 
                      y0 + line_length, int(x0 + 2 * cos30 * line_length)),
        ]
        for (rr, cc) in lines:
            img[rr,cc,:] = 0
        
    def draw_board_edges():
        red_lines = []
        blue_lines = []
        
        for i in range(board_size):
            x_top_start = int(left + i * 2 * line_length * cos30)
            # RED LINES:
            red_lines.append(
                draw.line(
                    top, x_top_start, 
                    int(top - line_length * sin30), int(x_top_start + line_length * cos30)
                )
            )
            red_lines.append(
                draw.line(
                    int(top - line_length * sin30), int(x_top_start + line_length * cos30), 
                    top, int(x_top_start + 2 * cos30 * line_length)
                )
            )
            y_bottom = int(top + (board_size-1) * (line_length + line_length * sin30)) + line_length
            x_bottom_start = int(left + (board_size-1) * line_length * cos30 + i * 2 * line_length * cos30)
            red_lines.append(
                draw.line(
                    y_bottom, x_bottom_start, 
                    int(y_bottom + line_length * sin30), int(x_bottom_start + line_length * cos30)
                )
            )
            red_lines.append(
                draw.line(
                    int(y_bottom + line_length * sin30), int(x_bottom_start + line_length * cos30), 
                    y_bottom, int(x_bottom_start + 2 * cos30 * line_length)
                )
            )
            
            # BLUES LINES:
            y_start = top + (1 + sin30) * line_length * i
            x_left_start = left + cos30 * line_length * i
            blue_lines.append(
                draw.line(
                    int(y_start), int(x_left_start), 
                    int(y_start + line_length), int(x_left_start)      
                )
            )
            if i < (board_size-1):
                blue_lines.append(
                    draw.line(
                        int(y_start + line_length), int(x_left_start), 
                        int(y_start + line_length * (1 + sin30)), int(x_left_start + line_length * cos30)      
                    )
                )
            x_right_start = left + (board_size * line_length * 2 * cos30) + cos30 * line_length * i
            blue_lines.append(
                draw.line(
                    int(y_start), int(x_right_start), 
                    int(y_start + line_length), int(x_right_start)      
                )
            )
            if i < (board_size-1):
                blue_lines.append(
                    draw.line(
                        int(y_start + line_length), int(x_right_start), 
                        int(y_start + line_length * (1 + sin30)), int(x_right_start + line_length * cos30)      
                    )
                )
                
        
        for (rr, cc) in red_lines:
            img[rr, cc, 0] = 255
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = 0
        
        for (rr, cc) in blue_lines:
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = 255
            
    def draw_piece(x, y, p):
        center_x = (left + line_length * cos30) + x * line_length * 2 * cos30 + y * line_length * cos30
        center_y = (top + line_length / 2) + y * (line_length + sin30 * line_length)
        rr, cc = draw.disk((int(center_y), int(center_x)), line_length/1.4)

        img[rr, cc, 0] = 255 if p == -1 else 0 # R
        img[rr, cc, 1] = 0 # G
        img[rr, cc, 2] = 255 if p == 1 else 0 # B
        
            
    for y in range(board_size):
        for x in range(board_size):
            draw_hexagon(
                int(left + x * 2 * line_length * cos30 + y * line_length * cos30), 
                int(top + y * (line_length + line_length * sin30)))

    draw_board_edges()

    for y in range(board_size):
        for x in range(board_size):
            p = position[y, x]
            if p != 0:
                draw_piece(x, y, p)
    
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def pop_random(arr: list):
    return arr.pop(randint(0, len(arr)-1))

def take_n_random(arr:list, n):
    taken = []
    for i in range(n):
        taken.append(pop_random(arr))
    return taken
    
def split_dataset(dataset, val_ratio = 0.2):
    black_wins = []
    white_wins = []
    for row in dataset:
        if row[-1] == 1:
            black_wins.append(row)
        else:
            white_wins.append(row)
    
    val = take_n_random(black_wins, int(len(black_wins) * val_ratio)) + \
            take_n_random(white_wins, int(len(white_wins) * val_ratio))
    
    train = black_wins + white_wins
    
    return train, val
    
def booleanize_positions(positions: np.ndarray):
    """
    2d representation of board position
    """
    g = np.zeros((positions.shape[0], 7, 14))
    for i in tqdm(range(positions.shape[0]), desc = "Booleanizing positions"):
        for y in range(7):
            for x in range(7):
                p = positions[i, y, x]
                if p == -1:
                    g[i, y, x] = 1
                elif p == 1:
                    g[i, y, x + 7] = 1
    return g

def booleanize_positions_3d(positions: np.ndarray):
    """
    3d representation of board position.
    """
    g = np.zeros((positions.shape[0], 2, 7, 7))
    for i in tqdm(range(positions.shape[0]), desc = "Booleanizing positions"):
        for y in range(7):
            for x in range(7):
                p = positions[i, y, x]
                if p == -1:
                    g[i, 0, y, x] = 1
                elif p == 1:
                    g[i, 1, y, x] = 1
    return g

def get_neighbour_lookup(board_size):
    global neighbours_lookups
    if board_size in neighbours_lookups:
        return neighbours_lookups[board_size]
    
    lookup = {}
    for y in range(board_size):
        for x in range(board_size):
            neighbours = []
            if x < board_size-1: # Right neighbour
                neighbours.append((y, x+1))
            if x > 0: # Left neighbour
                neighbours.append((y, x - 1))
            if y > 0: # Neighbours above
                neighbours.append((y-1, x))
                if x < board_size-1:
                    neighbours.append((y-1, x+1))
            if y < board_size-1: # Neighbours below
                neighbours.append((y+1, x))
                if x > 0:
                    neighbours.append((y+1, x-1))
            lookup[(y, x)] = neighbours
            
    neighbours_lookups[board_size] = lookup
    return lookup

def get_all_board_coordinates(board_size):
    global board_coordinates_lookups
    if board_size in board_coordinates_lookups:
        return board_coordinates_lookups[board_size]

    board_coordinates = []
    for y in range(board_size):
        for x in range(board_size):
            board_coordinates.append((y, x))
    
    board_coordinates_lookups[board_size] = board_coordinates
    
    return board_coordinates

def get_all_possible_connections(board_size):
    global all_possible_connections_lookup
    if board_size in all_possible_connections_lookup:
        return all_possible_connections_lookup[board_size]

    board_coordinates = get_all_board_coordinates(board_size)
    all_possible_connections = []
    for i in range(len(board_coordinates)):
        for j in range(i+1, len(board_coordinates)):
            connection = (*board_coordinates[i], *board_coordinates[j])
            all_possible_connections.append(connection)
    all_possible_connections_lookup[board_size] = all_possible_connections
    return all_possible_connections

def create_graph(board):
    graph = nx.Graph()
    board_size = board.shape[0]
    lookup = get_neighbour_lookup(board_size)
    
    # For each position; connect to nearby nodes
    for y in range(board_size):
        for x in range(board_size):
            # Add node
            piece = board[y,x]
            graph.add_node((y, x), piece = piece)

            # piece = 0 means empty and therefore it is not connected

            if piece == 0:
                continue

            # Connect to nearby nodes of same color
            for (ney, nex) in lookup[y, x]:
                if piece == board[ney, nex]:
                    graph.add_edge((y, x), (ney, nex))
    
    return graph

def display_as_graph(board):
    
    def get_piece_color(piece):
        match(piece):
            case -1: return "red"
            case 0: return "white"
            case 1: return "blue"
            
    G = create_graph(board)

    options = {
        "font_size": 0,
        "node_size": 100,
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    pos = {node: (node[1], -node[0]) for node in G.nodes}
    nodelist = [node for node in G.nodes]
    nodecolor = [get_piece_color(G.nodes[node]["piece"]) for node in nodelist]

    nx.draw_networkx(G, pos, nodelist = nodelist, node_color = nodecolor, **options)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

    plt.clf()

def get_board_at_n_moves_before_the_end(board_size, history, n, beginning_player):
    board = np.zeros((board_size, board_size), dtype=int)
    current_player = beginning_player
    if n == 0:
        selected_part_of_history = history
    else:
        selected_part_of_history = history[:-n]
    for mv in selected_part_of_history:
        y = mv // board_size
        x = mv % board_size
        board[y,x] = current_player
        current_player *= -1
    return board

def create_n_moves_before_the_end_dataset(file_name, board_size, n, beginning_player):
    history_file = open(file_name)
    games = set()
    boards = []
    winners = []
    for line in history_file:
        
        # To only save games once (make sure they are unique)
        if line in games:
            continue
        games.add(line)
        
        data = line.strip().split(",")
        data = [int(i) for i in data]
        
        winner = data[-1]
        history = data[:-1]
        board = get_board_at_n_moves_before_the_end(board_size, history, n, beginning_player)
        boards.append(board)
        winners.append(winner)
        
    history_file.close()
    return boards, winners

def save_dataset(boards:list[np.ndarray], winners, file_name):
    board_size = boards[0].shape[0]
    with open(file_name, "w+") as f:
        # Creating same CSV headers as kaggle dataset
        for i in range(board_size ** 2):
            y = i // board_size
            x = i % board_size
            f.write(f"cell_{y}_{x},")
        f.write("winner\n")
        
        # Populating dataset
        for i in range(len(boards)):
            board = boards[i]
            for j in board.flatten():
                f.write(f"{j},")
            f.write(f"{winners[i]}\n")