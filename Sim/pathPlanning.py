import numpy as np
from collections import deque
import matplotlib.pyplot as plt





#-- constants --

CELL_SIZE = 0.4 # meters

WORLD_X_MIN = -5 # meters
WORLD_X_MAX = 5 # meters
WORLD_Y_MIN = -5 # meters
WORLD_Y_MAX = 5 # meters


Z_HEIGHT = 1 #Z that will be input in the trajectory

# -- helper functions --

def build_grid(): #builds a grid of cells based on the world dimensions and cell size
    x_cells = int((WORLD_X_MAX - WORLD_X_MIN) / CELL_SIZE)
    y_cells = int((WORLD_Y_MAX - WORLD_Y_MIN) / CELL_SIZE)
    grid = np.zeros((x_cells, y_cells), dtype=int)
    return grid

def world_to_cell(wx, wy): #converts world coordinates to grid cell indexes
    cx = int((wx - WORLD_X_MIN) / CELL_SIZE)
    cy = int((wy - WORLD_Y_MIN) / CELL_SIZE)
    return cx, cy

def cell_to_world(cx, cy): #converts grid cell indexes back to world coordinates (center of the cell)
    wx = WORLD_X_MIN + cx * CELL_SIZE + CELL_SIZE / 2
    wy = WORLD_Y_MIN + cy * CELL_SIZE + CELL_SIZE / 2
    return wx, wy

def get_neighbors(cx, cy, grid): #returns the neighboring cells that are not walls
    neighbors = []
    nrow, ncol = grid.shape

    for dx in 0, 1, -1:
        for dy in 0, 1, -1:
            if (dx == 0 and dy == 0):
                continue #skip the current cell and walls
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < nrow and 0 <= ny < ncol:
                if grid[nx, ny] != -1:   # not a wall
                    neighbors.append((nx, ny))
    return neighbors



# -- brushfire functions --


def mark_walls(grid, map): #marks which cells contain walls
    
    #list of all the observed points
    xyz = map['xyz'] 

    nrow, ncol = grid.shape

    for pt in xyz:
        cx, cy = world_to_cell(pt[0], pt[1])
        if 0 <= cx < nrow and 0 <= cy < ncol:
            grid[cx, cy] = -1 #mark cell as occupied by a wall

    return grid

def brushfire_fill(grid, target_wx, target_wy): #comuputes distance to target for each cell using BFS
    
    target_cx, target_cy = world_to_cell(target_wx, target_wy)

    if grid[target_cx, target_cy] == -1:
        raise ValueError("Target cell is occupied by a wall")
    else: 
        #modified from 0 to 1 to distinguish from unvisited cells
        grid[target_cx, target_cy] = 1 



    queue = deque([(target_cx, target_cy)])

    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(grid.T, origin='lower', cmap='hot', interpolation='nearest')

    # BFS loop to populate the grid with distance to target 
    while queue:
        #removes the first cell in the queue and gets its coordinates
        cx, cy = queue.popleft()

        #distance from the traget is stored in the grid cell value
        current_distance = grid[cx, cy]

        for nx, ny in get_neighbors(cx, cy, grid):
            if grid[nx, ny] == 0: #unvisited cell
                grid[nx, ny] = current_distance + 1 #mark distance from target
                queue.append((nx, ny))
        img.set_data(grid.T)
        plt.pause(0.001)    # redraws the figure; lower = faster

    return grid

def create_path(grid, start_wx, start_wy): #creates a path from the start to the target by following the distance gradient
    start_cx, start_cy = world_to_cell(start_wx, start_wy)

    if grid[start_cx, start_cy] <= 0:
        raise ValueError("Start cell is occupied by a wall or unreachable")

    path = []
    cx, cy = start_cx, start_cy

    while grid[cx, cy] > 1: #while target has not been reached
        path.append(cell_to_world(cx, cy)) #add current cell to path
        neighbors = get_neighbors(cx, cy, grid)
        #find the neighbor with the smallest distance value
        cx, cy = min(neighbors, key=lambda n: grid[n[0], n[1]])

    path.append(cell_to_world(cx, cy)) #add target cell to path
    return path

def main():
    grid = build_grid()
    map = np.load('flight_map.npz') #load the map data

    mark_walls(grid, map) #mark the walls on the grid
    
    brushfire_fill(grid, 2, 2)# target
    path = create_path(grid, 0, 0)# start
    print(path)

if __name__ == "__main__":
    main()