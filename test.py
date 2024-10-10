
from utils import load_dataset, display_position, booleanize_positions


positions, winners = load_dataset("hex_games_1_000_000_size_7.csv", num_rows = 1000)

position = positions[1]
winner = winners[1]

print(position)
print(winner)
display_position(position)

print(booleanize_positions(positions[1:2]))