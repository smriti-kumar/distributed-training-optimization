import torch
from lib import codebook

cb = codebook.get_codebook("E8P12")
grid = cb.grid.float()

distances = torch.norm(grid - grid[0], dim=1)
min_distance = distances[distances > 1e-10].min()

directions = None
for i in range(grid.shape[0]):
    distances = torch.norm(grid - grid[i], dim=1)
    direction_mask = (distances - min_distance).abs() < 1e-5
    directions = grid[direction_mask] - grid[i]
    if len(directions) == 240:
        break

directions = torch.tensor(sorted(directions.tolist()), dtype=torch.float32)

neighbors_table = torch.full((grid.shape[0], 240), -1, dtype=torch.int32)
for i in range(grid.shape[0]):
    dist = torch.norm(grid - grid[i], dim=1)
    neighbor_mask = (dist - min_distance).abs() < 1e-5
    neighbor_indices = torch.where(neighbor_mask)[0].tolist()
    for j in neighbor_indices:
        direction = grid[j] - grid[i]
        dir_dist = torch.norm(directions - direction, dim=1)
        match = torch.where(dir_dist < 1e-5)[0]
        if len(match) > 0:
            neighbors_table[i, match[0].item()] = j

torch.save({"table": neighbors_table, "directions": directions}, "e8_2bit_neighbors.pt")