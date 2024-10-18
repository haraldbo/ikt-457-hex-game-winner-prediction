
from utils import load_dataset, display_as_graph, booleanize_positions_3d
import matplotlib.pyplot as plt
import numpy as np

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
    

positions, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = 10)

display_as_graph(positions[1])
create3d_board_representation(positions[1])
#position = positions[1]
#winner = winners[1]
#
#print(position)
#print(winner)
#display_position(position)
#
#print(booleanize_positions_v2(positions[1:2]))
#

