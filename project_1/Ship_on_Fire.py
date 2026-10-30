import random
import heapq
from collections import deque

# === UTIL ===
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def print_ship(grid, fire_grid, bot_pos, button_pos, title="Ship Map"):
    # display the grid
    grid_size = len(grid)
    print(f"\n-> {title} :")
    
    for r in range(grid_size):
        row_str = ""
        for c in range(grid_size):
            pos = (r, c)
            is_fire = fire_grid[r][c] if fire_grid else False
            
            if grid[r][c] == 'BLOCKED':
                row_str += "🧱"
            elif pos == bot_pos and is_fire:
                row_str += "💀"  # Bot caught in fire
            elif pos == bot_pos:
                row_str += "🤖"  # Bot location
            elif pos == button_pos:
                row_str += "🔘"  # Button location
            elif is_fire:
                row_str += "🔥"  # Fire location
            else:
                row_str += "⬜"  # Open empty path
        print(row_str)

def place_objects(grid, grid_size):
    # selects 3 distinct open cells for Bot, Button, and Fire at random
    # returns:
    # -> bot_pos: (r, c)
    # -> button_pos: (r, c)
    # -> fire_grid: 2D boolean array
    
    open_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r][c] == 'OPEN']
    
    # pick 3 distinct open cells at random
    bot_pos, button_pos, initial_fire_pos = random.sample(open_cells, 3)

    # initialize fire grid with starting fire location at t = 0
    fire_grid = [[False] * grid_size for _ in range(grid_size)]
    fire_grid[initial_fire_pos[0]][initial_fire_pos[1]] = True

    return bot_pos, button_pos, fire_grid
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

# === FIRE ===
def count_burning_neighbors(fire_grid, r, c, grid_size):
    k_count = 0
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if fire_grid[nr][nc]:
                k_count += 1
    return k_count

def advance_fire(grid, fire_grid, grid_size, q):
    # advances fire by 1 timestep based on probability 1 - (1 - q)^K
    next_fire_grid = [row[:] for row in fire_grid]
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r][c] == 'OPEN' and not fire_grid[r][c]:
                k = count_burning_neighbors(fire_grid, r, c, grid_size)
                if k > 0:
                    p = 1.0 - ((1.0 - q) ** k)
                    if random.random() < p:
                        next_fire_grid[r][c] = True
    return next_fire_grid
# =============

# === BOT 1 ===
def bot1(grid, grid_size, q):
    # plans shortest path, ignoring fire spread
    bot_pos, button_pos, fire_grid = place_objects(grid, grid_size)

    # retrieve initial fire position to avoid during initial path planning
    initial_fire_pos = [(r, c) for r in range(grid_size) for c in range(grid_size) if fire_grid[r][c]][0]

    #print_ship(grid, fire_grid, bot_pos, button_pos, title="Initial State")

    # Path Planning
    queue = deque([[bot_pos]]) #queue stores entire paths
    visited = {bot_pos, initial_fire_pos} #track all visited cells; do NOT return to intial pos or fire cell
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    planned_path = None

    while queue:
        path = queue.popleft()
        curr_r, curr_c = path[-1] #row and column of the last element

        if (curr_r, curr_c) == button_pos: #if the last element contains the button, you found a path
            planned_path = path
            break

        # constructs path by appending the neighbor coordinate
        # adds this extended path to the back of the queue
        for dr, dc in directions:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])

    # no valid path exists
    if planned_path is None:
        print("No valid path found.")
        return 'FAILED_NO_PATH', 0

    # Simulate Bot
    current_bot_pos = bot_pos

    for next_pos in planned_path[1:]:
        # bot moves to next cell
        current_bot_pos = next_pos

        # check if bot reaches button
        if current_bot_pos == button_pos:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title=f"Final State (SUCCESS)")
            return 'SUCCESS'

        # advance fire
        fire_grid = advance_fire(grid, fire_grid, grid_size, q)

        # check if fire reaches bot
        if fire_grid[current_bot_pos[0]][current_bot_pos[1]]:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title=f"Final State (CAUGHT BY FIRE)")
            return 'FAILED_CAUGHT_BY_FIRE'

    #print_ship(grid, fire_grid, current_bot_pos, button_pos, title=f"Final State (FAILED)")
    return 'FAILED_NO_PATH'
# =============

# === BOT 2 ===
def bot2(grid, grid_size, q):
    #plans shortest path to the button and avoids fire cells at every step
    bot_pos, button_pos, fire_grid = place_objects(grid, grid_size)

    #print_ship(grid, fire_grid, bot_pos, button_pos, title="Initial State")

    current_bot_pos = bot_pos

    # Path Planning
    while True:
        queue = deque([[current_bot_pos]])
        
        # treat current_bot_pos and all burning cells as visited
        visited = {current_bot_pos}
        for r in range(grid_size):
            for c in range(grid_size):
                if fire_grid[r][c]:
                    visited.add((r, c))

        planned_path = None

        while queue: # same as bot 1
            path = queue.popleft()
            curr_r, curr_c = path[-1]

            if (curr_r, curr_c) == button_pos:
                planned_path = path
                break

            for dr, dc in DIRECTIONS:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(path + [(nr, nc)])

        # Simulate Bot
        # Check if no valid path exists around current fire
        if planned_path is None or len(planned_path) < 2:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (NO PATH AVAILABLE)")
            return 'FAILED_NO_PATH'

        # move one step
        current_bot_pos = planned_path[1]

        # check if bot reaches button
        if current_bot_pos == button_pos:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (SUCCESS)")
            return 'SUCCESS'

        # fire advances
        fire_grid = advance_fire(grid, fire_grid, grid_size, q)

        # check if fire reaches bot
        if fire_grid[current_bot_pos[0]][current_bot_pos[1]]:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (CAUGHT BY FIRE)")
            return 'FAILED_CAUGHT_BY_FIRE'
# =============

# === BOT 3 ===
def bot3(grid, grid_size, q):
    # attempts to find a path avoiding fire AND fire-adjacent cells
    # if impossible, it falls back to a path avoiding only fire cells
    # executes the first step of the chosen path
    
    bot_pos, button_pos, fire_grid = place_objects(grid, grid_size)
    #print_ship(grid, fire_grid, bot_pos, button_pos, title="Initial State")

    current_bot_pos = bot_pos

    # run bfs given a set of blocked cells
    def find_path(start, target, blocked_cells):
        queue = deque([[start]])
        visited = set(blocked_cells)
        visited.add(start)

        while queue:
            path = queue.popleft()
            curr_r, curr_c = path[-1]

            if (curr_r, curr_c) == target:
                return path

            for dr, dc in DIRECTIONS:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(path + [(nr, nc)])
        return None

    # simulation loop
    while True:
        # identify all fire cells AND fire-adjacent cells
        current_fire_cells = set()
        fire_adjacent_cells = set()

        for r in range(grid_size):
            for c in range(grid_size):
                if fire_grid[r][c]:
                    current_fire_cells.add((r, c))
                    for dr, dc in DIRECTIONS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if grid[nr][nc] == 'OPEN':
                                fire_adjacent_cells.add((nr, nc))

        # avoiding fire AND fire-adjacent cells
        strict_obstacles = (current_fire_cells | fire_adjacent_cells) - {button_pos}
        planned_path = find_path(current_bot_pos, button_pos, strict_obstacles)

        # avoiding ONLY fire cells
        if planned_path is None:
            planned_path = find_path(current_bot_pos, button_pos, current_fire_cells)

        # SIMULATE
        # if no path exists under either tier, bot is trapped
        if planned_path is None or len(planned_path) < 2:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (NO PATH AVAILABLE)")
            return 'FAILED_NO_PATH'

        # move bot one step
        current_bot_pos = planned_path[1]

        # check if bot reaches before fire advances
        if current_bot_pos == button_pos:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (SUCCESS)")
            return 'SUCCESS'

        # fire advances
        fire_grid = advance_fire(grid, fire_grid, grid_size, q)

        # check if fire caught the bot
        if fire_grid[current_bot_pos[0]][current_bot_pos[1]]:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (CAUGHT BY FIRE)")
            return 'FAILED_CAUGHT_BY_FIRE'
# =============

# === BOT 4 ===
def distances_to_button(grid, button_pos, grid_size):
    # computes bfs distances from every open cell to the button
    # heuristic for A* search
    
    distances = {}
    queue = deque([(button_pos, 0)])
    visited = {button_pos}

    while queue:
        (curr_r, curr_c), dist = queue.popleft()
        distances[(curr_r, curr_c)] = dist

        for dr, dc in DIRECTIONS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

    return distances

def compute_fire_distances(grid, fire_grid, grid_size):
    # bfs calculating shortest distance from any cell to the nearest burning cell.
    
    fire_distances = {}
    queue = deque()
    visited = set()

    for r in range(grid_size):
        for c in range(grid_size):
            if fire_grid[r][c]:
                queue.append(((r, c), 0))
                visited.add((r, c))

    while queue:
        (curr_r, curr_c), dist = queue.popleft()
        fire_distances[(curr_r, curr_c)] = dist

        for dr, dc in DIRECTIONS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr][nc] == 'OPEN' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

    return fire_distances

def compute_cell_danger(r, c, grid, fire_grid, fire_distances, q, grid_size, danger_weight=10.0):
    # calculates cell danger using fire spread probability & distance to fire
    if fire_grid[r][c]:
        return float('inf')
    
    # direct catch prob for cells adjacent to fire: 1 - (1 - q)^K
    k = count_burning_neighbors(fire_grid, r, c, grid_size)
    if k > 0:
        p_catch = 1.0 - ((1.0 - q) ** k)
        return danger_weight * p_catch

    dist_to_fire = fire_distances.get((r, c), float('inf'))
    if dist_to_fire == float('inf'):
        return 0.0

    # risk based on flammability q and proximity
    # proximity decay for cells further away: p_max / (1 + dist_to_fire)
    p_base = 1.0 - (1.0 - q)
    return danger_weight * (p_base / (1.0 + dist_to_fire))

def bot4(grid, grid_size, q, danger_weight=10.0):
    # executes Bot 4 simulation using A* pathfinding. re-plans at every step by 
    # evaluating exact step costs (1 + danger) and precomputed bfs heuristic.
    
    bot_pos, button_pos, fire_grid = place_objects(grid, grid_size)
    #print_ship(grid, fire_grid, bot_pos, button_pos, title="Initial State")

    # precompute wall-aware bfs distances from all cells to button (A* heuristic)
    h_dist = distances_to_button(grid, button_pos, grid_size)

    # if button not reachable --> fail
    if bot_pos not in h_dist:
        print_ship(grid, fire_grid, bot_pos, button_pos, title="Final State (NO PATH AVAILABLE)")
        return 'FAILED_NO_PATH'

    current_bot_pos = bot_pos

    # simulation loop
    while True:
        # compute fire distance map across ship
        fire_distances = compute_fire_distances(grid, fire_grid, grid_size)

        # run A* pathfinding from current position to button
        # priority queue stores: (f_score, g_score, current_cell, path)
        # automatically sorts the heap by the first element
        open_set = []
        heapq.heappush(open_set, (h_dist[current_bot_pos], 0.0, current_bot_pos, [current_bot_pos]))
        # tracking the best known g-score for every cell
        g_scores = {current_bot_pos: 0.0}
        planned_path = None

        while open_set:
            f_score, g_score, curr, path = heapq.heappop(open_set)

            if curr == button_pos:
                planned_path = path
                break

            # if we found a cheaper route to curr earlier,
            # this popped element is an outdated duplicate and we can skip it
            if g_score > g_scores.get(curr, float('inf')):
                continue

            for dr, dc in DIRECTIONS:
                nr, nc = curr[0] + dr, curr[1] + dc
                neighbor = (nr, nc)

                if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr][nc] == 'OPEN':
                    if fire_grid[nr][nc]:
                        continue  # do not enter cells currently on fire

                    danger = compute_cell_danger(nr, nc, grid, fire_grid, fire_distances, q, grid_size, danger_weight)
                    step_cost = 1.0 + danger
                    tentative_g = g_score + step_cost

                    # if new path to neighbor is cheaper than any previously recorded path
                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        # update the dictionary with the new best score
                        g_scores[neighbor] = tentative_g
                        h_cost = h_dist.get(neighbor, float('inf'))
                        if h_cost != float('inf'):
                            f_neighbor = tentative_g + h_cost
                            # push the new tuple onto the heap
                            heapq.heappush(open_set, (f_neighbor, tentative_g, neighbor, path + [neighbor]))

        # SIMULATION
        # if no valid path found, bot is trapped
        if planned_path is None or len(planned_path) < 2:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (NO PATH AVAILABLE)")
            return 'FAILED_NO_PATH'

        # execute first step of A* path
        current_bot_pos = planned_path[1]

        # check if button reached
        if current_bot_pos == button_pos:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (SUCCESS)")
            return 'SUCCESS'

        # advance fire by 1 timestep
        fire_grid = advance_fire(grid, fire_grid, grid_size, q)

        # check if caught by fire
        if fire_grid[current_bot_pos[0]][current_bot_pos[1]]:
            #print_ship(grid, fire_grid, current_bot_pos, button_pos, title="Final State (CAUGHT BY FIRE)")
            return 'FAILED_CAUGHT_BY_FIRE'
# =============

# === EXECUTION ===
if __name__ == "__main__":
    d = 20
    q = 0.50
    
    ship = generate_ship(d)

    bot_4 = bot4(ship, d, q)
    print(f"Outcome of Bot 4: {bot_4}")