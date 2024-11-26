"""
Captures games from bot playing against bot on https://lutanho.net/play/hex.html
"""
from PIL import ImageGrab
import pyautogui as gui
import numpy as np
import time
import networkx as nx
from networkx import has_path
from pathlib import Path

config = {
    # Coordinates: (y, x)
    "top": (225, 854), # Center of the topmost hexagon
    "left": (564, 652), # Center of the leftmost hexagon
    "new_game_button": (647, 1188),
    "board_size": 9,
    "blue_level_1": (1, 2), # radio button blue level 1
    "red_level_1": (1, 2) # radio button red level 1
}

def test_fps():
    fps = 0
    start = time.time_ns()
    while time.time_ns() - start < 1_000_000_000:
        fps += 1 
        capture_board()
    print(fps)

def test_hex_coordinates():
    """
    Returns screen coordinates of all hexagons
    """
    board_size = config["board_size"]
    dy = (config["left"][0] - config["top"][0])/8
    dx = (config["top"][1] - config["left"][1])/8

    y0 = config["top"][0]
    x0 = config["top"][1]

    board_screen_coordiantes = {}

    for y in range(board_size):
        for x in range(board_size):
            y_screen = (y0 + y * dy) + dy * x
            x_screen = (x0 - y * dx) + dx * x
            board_screen_coordiantes()
            gui.moveTo(x_screen, y_screen)
            time.sleep(0.1)

def piece_from_pixel_color(rgb):
    if rgb == (255, 255, 255): # White
        return 0
    elif rgb == (255, 0, 0): # Red
        return -1
    elif rgb == (0, 0, 255): # Blu
        return 1
    else:
        raise ValueError(f"Unknown color: {rgb}")

def create_empty_board(board_size):
    return [[0 for i in range(board_size)] for j in range(board_size)]    

def capture_board():
    board_size = config["board_size"]
    board = create_empty_board(board_size)
    dy = (config["left"][0] - config["top"][0])/8
    dx = (config["top"][1] - config["left"][1])/8

    y0 = config["top"][0]
    x0 = config["top"][1]

    screen_img = ImageGrab.grab()
    #print(screen_img.size)
    for y in range(board_size):
        for x in range(board_size):
            y_screen = (y0 + y * dy) + dy * x
            x_screen = (x0 - y * dx) + dx * x
            #gui.moveTo(x_screen, y_screen)
            board[y][x] = piece_from_pixel_color(screen_img.getpixel((x_screen, y_screen)))

    return board
    
def select_random_ai_level():
    pass

def press_new_game_button():
    (y, x) = config["new_game_button"]
    gui.leftClick(x, y)
    time.sleep(0.1)

def add_piece(board_graph: nx.Graph, y, x, piece):
    board_size = config["board_size"]
   
    board_graph.add_node((y, x), piece = piece)
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

def get_winner(board_graph):
    """
    Returns player that has won. Returns 0 if no winner.
    
    -1 needs to make a connection from first row to last row
    1 needs to make a connection from first column to last column

    """
    board_size = config["board_size"]
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
    
def extract_move(previous_board, current_board):
    board_size = config["board_size"]
    for y in range(board_size):
        for x in range(board_size):
            if previous_board[y][x] != current_board[y][x]:
                return y, x, current_board[y][x]
    raise ValueError("Could not find new move from boards")

def count_moves_done(board):
    count = 0
    board_size = config["board_size"]
    for y in range(board_size):
        for x in range(board_size):
            if board[y][x] != 0:
                count += 1
    return count

def create_move_history(boards):
    moves = []
    board_size = config["board_size"]
    for i in range(1, len(boards)):
        board1 = boards[i-1]
        board2 = boards[i]
        (y, x, _) = extract_move(board1, board2)
        moves.append(y * board_size + x)
    return moves

def capture_game():
    board_size = config["board_size"]
    boards = [create_empty_board(board_size)]
    winner = 0
    game_graph = create_empty_board_graph(board_size)

    board = capture_board()
    press_new_game_button()

    timeout_time_ns = time.time_ns() + 1_000_000_000 * 5
    while count_moves_done(capture_board()) not in [0, 1]:
        if time.time_ns() > timeout_time_ns:
            raise TimeoutError("Timed out waiting for board to reset")
        time.sleep(0.01)

    timeout_time_ns = time.time_ns() + 1_000_000_000 * 2
    while not winner:
        board = capture_board()
        if board in boards:
            if time.time_ns() > timeout_time_ns:
                raise TimeoutError("Timed out waiting for new move to be done")
            
            continue
        
        timeout_time_ns = time.time_ns() + 1_000_000_000 * 2
        boards.append(board)
        
        # Extract the new move by comparing to previous board 
        (y, x, piece) = extract_move(boards[-2], board)
        
        # Add move to the graph and check if player has won
        add_piece(game_graph, y, x, piece)
        winner = get_winner(game_graph)
    return create_move_history(boards), winner
      
def append_to_dataset_file(file_name, history, winner):
    with open(Path(__file__).parent / file_name, mode = "a") as file:
        for x in history:
            file.write(f"{x},")
        file.write(f"{winner}\n")
def start():
    i = 0
    dataset_size = 100
    while True:
        try:
            history, winner = capture_game()
            append_to_dataset_file("9x9_games.txt", history, winner)
            select_random_ai_level()
            time.sleep(3)
            i += 1
            if i == dataset_size:
                exit(0)
        except TimeoutError as e:
            print(e)

print("Starting in 5 seconds")
time.sleep(5)
#test_fps()
start()