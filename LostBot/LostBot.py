import random
import math
from collections import deque

# === UTIL ===
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def place_objects(grid, grid_size):
    # selects 2 distinct open cells for Locator and Lost Bot
    open_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r][c] == 'OPEN']
    # pick 2 distinct open cells at random
    locator_bot, lost_bot = random.sample(open_cells, 2)
    return locator_bot, lost_bot, open_cells
# ============

# === THE SHIP ===
def count_open_neighbors(grid, r, c, grid_size): #r = row, c = col
    count = 0
    # checks up, down, left, and right for open cells, increments count if true
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if grid[nr][nc] == 'OPEN':
                count += 1
                
    return count

def get_blocked_neighbors(grid, r, c, grid_size):
    # returns a list of (row, col) tuples for adjacent blocked cells.
    blocked = []
    # checks up, down, left, and right for blocked cells, appends to a blocked arr if true
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if grid[nr][nc] == 'BLOCKED':
                blocked.append((nr, nc))
                
    return blocked

def find_dead_ends(grid, grid_size):
    # finds all open cells that have exactly 1 open neighbor.
    dead_ends = []

    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r][c] == 'OPEN' and count_open_neighbors(grid, r, c, grid_size) == 1:
                dead_ends.append((r, c))
    return dead_ends

def generate_ship(d):
    # create a DxD grid with all cells blocked
    grid = [['BLOCKED'] * d for _ in range(d)]
    
    # create an open cell at a random row and col
    start_r = random.randint(1, d - 2)
    start_c = random.randint(1, d - 2)
    grid[start_r][start_c] = 'OPEN'
    
    # create hallways
    while True:
        candidates = []
        for r in range(d):
            for c in range(d):
                # identify all currently blocked cells that have exactly one open neighbor
                if grid[r][c] == 'BLOCKED' and count_open_neighbors(grid, r, c, d) == 1:
                    candidates.append((r, c))

        # breaks when no blocked cells have exactly one open neighbor
        if not candidates:
            break

        # of these currently blocked cells with exactly one open neighbor, pick one at random.
        open_r, open_c = random.choice(candidates)
        # open the selected cell.
        grid[open_r][open_c] = 'OPEN'
        
    # identify dead ends
    dead_ends = find_dead_ends(grid, d)
    target_dead_ends = len(dead_ends) // 2
    
    # reduce dead ends by at least half
    while len(dead_ends) > target_dead_ends:
        current_dead_end = random.choice(dead_ends)
        r, c = current_dead_end
        
        blocked_neighbors = get_blocked_neighbors(grid, r, c, d)
        if blocked_neighbors:
            wall_r, wall_c = random.choice(blocked_neighbors)
            grid[wall_r][wall_c] = 'OPEN'
            
        # update dead_ends
        dead_ends = find_dead_ends(grid, d)
        
    return grid
# ================

# === DETECTORS ===
def detector_a(locator_pos, lost_pos, k):
    # Delivers a ping if the bot is within a (2k+1)x(2k+1) square (Chebyshev distance <= k).
    if locator_pos == lost_pos:
        return "FOUND"
    
    r1, c1 = locator_pos
    r2, c2 = lost_pos
    # chebyshev distance for a square grid
    chebyshev_dist = max(abs(r1 - r2), abs(c1 - c2))
    
    return chebyshev_dist <= k

def detector_b(locator_pos, lost_pos, alpha):
    # delivers a ping with probability P = e^(-alpha * (dist(a, b) - 1)) using Euclidean distance.
    if locator_pos == lost_pos:
        return "FOUND"
        
    r1, c1 = locator_pos
    r2, c2 = lost_pos
    
    # calculate Euclidean distance
    dist = math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    # calculate ping probability
    prob = math.exp(-alpha * (dist - 1))
    prob = max(0.0, min(1.0, prob))  # Ensure valid probability range [0, 1]
    
    return random.random() < prob
# =================

# === BASELINE ===
def bfs(grid, start, target, grid_size):
    # finds the shortest path of open cells from start to target using BFS
    if start == target:
        return [start]
        
    queue = deque([[start]])
    visited = {start}
    
    while queue:
        path = queue.popleft()
        curr_r, curr_c = path[-1]
        
        if (curr_r, curr_c) == target:
            return path
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
                    
    return []

def baseline_a(grid, grid_size, k):
    locator_bot, lost_bot, open_cells = place_objects(grid, grid_size)
    possible_locations = set(open_cells)
    possible_locations.discard(locator_bot) #prevents bfs from picking the current cell

    move_count = 0
    detector_count = 0
    
    while True:
        # runs detector
        detector_count += 1
        result = detector_a(locator_bot, lost_bot, k)
        
        if result == "FOUND":
            return {"status": "FOUND", "moves": move_count, "detectors": detector_count}

        possible_locations.discard(locator_bot) #prevents bfs from picking the current cell

        # update possible locations based on ping vs no ping
        r_loc, c_loc = locator_bot
        if result:  # PING: Bot is within (2k+1)x(2k+1) square
            possible_locations = {
                (r, c) for r, c in possible_locations 
                if max(abs(r - r_loc), abs(c - c_loc)) <= k
            }
        else:  # NO PING: Bot is outside (2k+1)x(2k+1) square
            possible_locations = {
                (r, c) for r, c in possible_locations 
                if max(abs(r - r_loc), abs(c - c_loc)) > k
            }
            
        if not possible_locations:
            return {"status": "FAILED", "moves": move_count, "detectors": detector_count}
            
        # move to the nearest cell that might contain LostBot (breaking ties at random)
        best_target = min(
            possible_locations, 
            key=lambda cell: (len(bfs(grid, locator_bot, cell, grid_size)), random.random())
        )
        
        path = bfs(grid, locator_bot, best_target, grid_size)
        if len(path) > 1:
            locator_bot = path[1]  # take one step along the path
            move_count += 1
        else:
            locator_bot = best_target

def baseline_b(grid, grid_size, alpha):
    locator_bot, lost_bot, open_cells = place_objects(grid, grid_size)
    
    # initialize uniform probabilistic belief over all open cells
    num_cells = len(open_cells)

    beliefs = {cell: 1.0 / num_cells
               for cell in open_cells
               if cell != locator_bot}
    
    move_count = 0
    detector_count = 0
    
    while True:
        # run detector
        detector_count += 1
        result = detector_b(locator_bot, lost_bot, alpha)
        
        if result == "FOUND":
            return {"status": "FOUND", "moves": move_count, "detectors": detector_count}
        
        beliefs.pop(locator_bot, None)
            
        # update knowledgebase (Bayes' Rule Update)
        r_loc, c_loc = locator_bot
        new_beliefs = {}
        total_prob = 0.0
        
        for cell in beliefs:
            r_b, c_b = cell
            dist = math.sqrt((r_loc - r_b)**2 + (c_loc - c_b)**2)
            p_ping = max(0.0, min(1.0, math.exp(-alpha * (dist - 1))))
            
            likelihood = p_ping if result else (1.0 - p_ping)
            unnormalized = likelihood * beliefs[cell]
            new_beliefs[cell] = unnormalized
            total_prob += unnormalized
            
        # normalize beliefs
        if total_prob > 0:
            beliefs = {cell: prob / total_prob for cell, prob in new_beliefs.items()}
            
        # move to the cell most likely to contain LostBot (breaking ties by distance, then random)
        max_prob = max(beliefs.values())
        top_cells = [cell for cell, prob in beliefs.items() if math.isclose(prob, max_prob, rel_tol=1e-9)]
        
        best_target = min(
            top_cells,
            key=lambda cell: (len(bfs(grid, locator_bot, cell, grid_size)), random.random())
        )
        
        path = bfs(grid, locator_bot, best_target, grid_size)
        if len(path) > 1:
            locator_bot = path[1]
            move_count += 1
        else:
            locator_bot = best_target
# ================

# === IMPROVED ===
# add improved versions of locator bot A and B here

# ================

# === EXECUTION ===
if __name__ == "__main__":
    d = 15
    k = 3
    alpha = 0.5
    ship = generate_ship(d)

    total_moves_a = 0
    total_detectors_a = 0
    total_moves_b = 0
    total_detectors_b = 0

    for i in range(1, 11):
        base_a = baseline_a(ship, d, k)
        base_b = baseline_b(ship, d, alpha)

        total_moves_a += base_a["moves"]
        total_detectors_a += base_a["detectors"]
        total_moves_b += base_b["moves"]
        total_detectors_b += base_b["detectors"]

        avg_moves_a = total_moves_a / 10
        avg_detectors_a = total_detectors_a / 10
        avg_moves_b = total_moves_b / 10
        avg_detectors_b = total_detectors_b / 10
        
        print(f"Baseline A- {i}th Run: {base_a}")
        print(f"Baseline B- {i}th Run: {base_b}")

#Run this simulation for the improved version and compare the averages across a range of k [1-d] and alpha [0.05-0.5]
print(f"Average Moves A: {avg_moves_a:.2f}")
print(f"Average Detectors A: {avg_detectors_a:.2f}")
print(f"Average Moves B: {avg_moves_b:.2f}")
print(f"Average Detectors B: {avg_detectors_b:.2f}")