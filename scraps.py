
from utils import load_dataset, display_as_graph, booleanize_positions_3d, display_position, get_board_at_n_moves_before_the_end, create_n_moves_before_the_end_dataset, save_dataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def write_csv_example():
    results_csv = open("results.csv", mode = "w+")
    results_csv.write(f"{epoch},{train_accuracy},{test_accuracy}")
    results_csv.close()
    
def create3d_board_representation(position):
    """
    Display a (2, 7, 7) representation of the booleanized board
    
    This method was cooked by modifying 
    https://matplotlib.org/stable/gallery/mplot3d/voxels_numpy_logo.html#sphx-glr-gallery-mplot3d-voxels-numpy-logo-py
    
    """
    position = booleanize_positions_3d(position)
    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    # build up the numpy logo
    n_voxels = np.zeros((2, 7, 7), dtype=bool)
    for z in range(position.shape[0]):
        for y in range(position.shape[1]):
            for x in range(position.shape[2]):
                n_voxels[z, y, x] = position[z, y, x] == 1
    facecolors = np.where(n_voxels, '#FFFFFFFF', '#000000FF')
    edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
    filled = np.ones(n_voxels.shape)

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    ax.set_aspect('equal')

    plt.show()



def create_table_of_boardv2(board):
    new_board = np.zeros((board.shape[0], board.shape[1] * 2), dtype=int)
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y, x] == 1:
                new_board[y, x] = 1
            
            elif board[y,x] == -1:
                new_board[y, x + board.shape[1]] = 1
    
    for y in range(new_board.shape[0]):
        print("&".join(new_board[y].astype(str)))

def get_unique_games():
    games = set()
    with open("9x9_games.txt") as file:
        for line in file.readlines():
            games.add(line)
    
    return games


boards, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = 10)
#
#display_as_graph(positions[1])
#create3d_board_representation(positions[1])
#position = positions[1]
#winner = winners[1]
#
#print(position)
#print(winner)
#display_position(position)
#
#print(booleanize_positions_v2(positions[1:2]))
#
board1 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, 0],
])

board2 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])


board3 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])


board = np.array([
    [0, -1, 0, 0],
    [0, -1, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
])

#create_table_of_boardv2(boards[1])
#display_position(board3)



boards, winners = create_n_moves_before_the_end_dataset("9x9_games_red_26112024.txt", 9, 2, -1)

save_dataset(boards, winners, Path(__file__).parent / "dataset"/ "dataset9x9_2moves.csv")

boards, winners = load_dataset("dataset9x9_2moves.csv")

print("Num unique games:", len(boards))