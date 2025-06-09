# Energy-Aware A* Path Planning for AGVs using State-Based Energy Modeling

import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# --- Constants from experimental data ---
P_CTRL = 63.19      # Active control power (W)
P_DRIVE = 55.15     # Drive power (W)
P_LHD = 8.67        # Load handling power (W)
T_DRIVE = 1.0       # Time per driving step (s)
T_LOAD = 5.0        # Time per load/unload (s)
ENERGY_CAPACITY = 480  # Battery capacity (Wh)
BATTERY_THRESHOLD = 10  # % battery threshold for charging trigger
LOAD_PENALTY = 5.0  # Drive power penalty when loaded

# --- State power model ---
STATE_POWER = {
    "drive_empty": [P_CTRL, P_DRIVE, 0],
    "drive_loaded": [P_CTRL, P_DRIVE + LOAD_PENALTY, 0],
    "acceleration_empty": [P_CTRL, P_DRIVE + 10, 0],
    "acceleration_loaded": [P_CTRL, P_DRIVE + 10 + LOAD_PENALTY, 0],
    "deceleration_empty": [P_CTRL, P_DRIVE + 5, 0],
    "deceleration_loaded": [P_CTRL, P_DRIVE + 5 + LOAD_PENALTY, 0],
    "pickup": [P_CTRL, 0, P_LHD],
    "dropoff": [P_CTRL, 0, P_LHD],
    "standby": [7.5, 11.65, 0],
}

# --- Heuristic for A* ---
def heuristic(a, b, load_status):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    distance = max(dx, dy)

    # Assume a best-case: only straight-line driving (constant speed)
    state = "drive_loaded" if load_status else "drive_empty"
    power_components = STATE_POWER[state]
    total_power = sum(power_components)

    time = T_DRIVE * distance
    energy_wh = (total_power * time) / 3600
    return energy_wh


# --- Neighbor generation (8 directions) ---
def neighbors(grid, node):
    x, y = node
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
            yield (nx, ny), dx, dy

# --- Determine current AGV state ---
def determine_state(prev, current, load_status, prev_dir=None, mode="drive"):
    dx, dy = current[0] - prev[0], current[1] - prev[1]

    if dx == 0 and dy == 0:
        return "standby"
    elif mode == "pickup":
        return "pickup"
    elif mode == "dropoff":
        return "dropoff"
    elif prev_dir is not None:
        # Compare direction change for acceleration/deceleration
        if (dx, dy) != prev_dir:
            return "acceleration_loaded" if load_status else "acceleration_empty"
        else:
            return "drive_loaded" if load_status else "drive_empty"
    else:
        # First move — assume it's acceleration
        return "acceleration_loaded" if load_status else "acceleration_empty"


# --- Calculate energy for a given state ---
def calculate_state_energy(state, duration=T_DRIVE):
    P_ctrl, P_drive, P_lhd = STATE_POWER[state]
    total_power = P_ctrl + P_drive + P_lhd
    return (total_power * duration) / 3600  # in Wh

# --- A* Algorithm with Energy Awareness ---
def astar_energy_optimized(grid, start, goal, battery_level, load_status=False):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal, load_status), 0, start, (0, 0), battery_level))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, load_status)}
    turns = 0
    steps = 0
    path_states = []

    while open_list:
        _, current_g, current, last_dir, battery_level = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], battery_level, steps, turns, path_states

        for neighbor, dx, dy in neighbors(grid, current):
            state = determine_state(current, neighbor, load_status, last_dir)
            energy_cost = calculate_state_energy(state)

            tentative_g = current_g + energy_cost
            remaining_battery = battery_level - (energy_cost / ENERGY_CAPACITY * 100)
            if remaining_battery < BATTERY_THRESHOLD:
                continue  # skip if battery too low
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal, load_status)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g, neighbor, (dx, dy), remaining_battery))
                steps += 1
                if (dx, dy) != last_dir:
                    turns += 1
                path_states.append(state)

    return [], battery_level, steps, turns, path_states

import numpy as np
from scipy.interpolate import CubicSpline

def buffered_is_line_clear(grid, p1, p2, margin=1, steps=50):
    """Check if a line from p1 to p2 is free of obstacles, with safety margin."""
    x1, y1 = p1
    x2, y2 = p2
    for t in np.linspace(0, 1, steps):
        x = int(round(x1 * (1 - t) + x2 * t))
        y = int(round(y1 * (1 - t) + y2 * t))
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                    if grid[nx][ny] != 0:
                        return False
    return True

def shortcut_smooth(grid, path, margin=1):
    """Simplify the path by removing unnecessary waypoints while maintaining safe distance."""
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if buffered_is_line_clear(grid, path[i], path[j], margin=margin):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed

def is_point_safe(grid, x, y, margin=1):
    """Check if a point is safely away from any obstacles."""
    xi, yi = int(round(x)), int(round(y))
    for dx in range(-margin, margin + 1):
        for dy in range(-margin, margin + 1):
            nx, ny = xi + dx, yi + dy
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):  # Note: grid[y][x]
                if grid[ny][nx] != 0:
                    return False
            else:
                return False  # Treat out-of-bounds as unsafe
    return True

def smooth_path(path, grid=None, margin=1, nudge_distance=0.1, max_attempts=16):
    """Smooth a path using cubic splines and nudge points away from obstacles if needed."""
    if len(path) < 3:
        return path

    x = [p[1] for p in path]
    y = [p[0] for p in path]
    spline_x = CubicSpline(range(len(x)), x, bc_type='natural')
    spline_y = CubicSpline(range(len(y)), y, bc_type='natural')
    smoothed_x = spline_x(np.linspace(0, len(x) - 1, 100))
    smoothed_y = spline_y(np.linspace(0, len(y) - 1, 100))

    smoothed = []
    for sx, sy in zip(smoothed_x, smoothed_y):
        if grid is None or is_point_safe(grid, sx, sy, margin):
            smoothed.append((sy, sx))
            continue

        # Try multiple radii of nudging to find a safe nearby point
        found_safe = False
        for radius in np.linspace(nudge_distance, 1.0, 5):  # Expand radius
            for angle in np.linspace(0, 2 * np.pi, max_attempts, endpoint=False):
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                nx, ny = sx + dx, sy + dy
                if is_point_safe(grid, nx, ny, margin):
                    smoothed.append((ny, nx))
                    found_safe = True
                    break
            if found_safe:
                break

        if not found_safe:
            # Optionally keep or skip unsafe point — here we try to keep it
            smoothed.append((sy, sx))  # Force add, even if not safe

    return smoothed



# --- Total Energy Calculation ---
def calculate_total_energy(path_states):
    energy = 0
    for state in path_states:
        energy += calculate_state_energy(state)
    return energy  # Total in Wh

def calculate_total_time(path_states):
    total_time = 0
    for state in path_states:
        if "pickup" in state or "dropoff" in state:
            total_time += T_LOAD
        elif state == "standby":
            total_time += 1  # or whatever idle time
        else:
            total_time += T_DRIVE
    return total_time

# --- Visualization ---
import numpy as np
import matplotlib.pyplot as plt
import heapq


def visualize_path(grid, path, energy, steps, turns, time_taken):
    grid_display = np.array(grid, dtype=str)

    for (x, y) in path:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < len(grid) and 0 <= yi < len(grid[0]):
            grid_display[xi][yi] = '.'

    sx, sy = map(lambda v: int(round(v)), path[0])
    gx, gy = map(lambda v: int(round(v)), path[-1])

    if 0 <= sx < len(grid) and 0 <= sy < len(grid[0]):
        grid_display[sx][sy] = 'S'
    if 0 <= gx < len(grid) and 0 <= gy < len(grid[0]):
        grid_display[gx][gy] = 'G'

    print("\nPath Grid:")
    for row in grid_display:
        print(" ".join(row))

    print("\n--- Path Statistics ---")
    print(f"Total Steps: {steps}")
    print(f"Total Turns: {turns}")
    print(f"Total Energy Consumed: {energy:.2f} Wh")
    print(f"Estimated Time Taken: {time_taken:.2f} s")


def visualize_path_with_matplotlib(grid, path, energy, steps, turns, time_taken):
    grid_array = np.array(grid)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Show the grid (0=free=white, 1=obstacle=black)
    ax.imshow(grid_array, cmap='Greys', origin='upper')

    # Unpack path points correctly: path tuples are (x,y), so split into x_coords, y_coords
    x_coords, y_coords = zip(*path)
    ax.plot(y_coords, x_coords, marker='o', color='blue', linewidth=2, label='Path')
    ax.scatter(y_coords[0], x_coords[0], color='green', s=100, label='Start')
    ax.scatter(y_coords[-1], x_coords[-1], color='red', s=100, label='Goal')

    ax.set_title("Energy-Aware A* Path")
    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.set_xticklabels(np.arange(len(grid[0])))
    ax.set_yticklabels(np.arange(len(grid)))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    ax.legend()
    ax.set_aspect('equal')

    plt.text(0, len(grid), f"Steps: {steps}, Turns: {turns}, Energy: {energy:.2f} Wh, Time: {time_taken:.2f} s",
             fontsize=10, ha='left')

    plt.tight_layout()
    plt.show()
def generate_high_res_grid(grid, factor=5):
    """Return a high-res version of the grid with expanded obstacles."""
    import numpy as np
    grid = np.array(grid)
    high_res = np.kron(grid, np.ones((factor, factor), dtype=int))

    return high_res



def main():
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]

    factor = 5  # Increase resolution by 5x
    high_res = generate_high_res_grid(grid, factor)

    start = (0, 0)
    goal = (9, 9)
    start_hr = (start[0] * factor, start[1] * factor)
    goal_hr = (goal[0] * factor, goal[1] * factor)

    battery_level = 100  # Start with full battery

    path_hr, remaining_battery, steps, turns, path_states = astar_energy_optimized(
        high_res, start_hr, goal_hr, battery_level, load_status=False)

    shortcut_path = shortcut_smooth(high_res, path_hr, margin=2)

    # Apply spline smoothing with nudging away from obstacles
    smoothed_path = smooth_path(shortcut_path, grid=high_res, margin=2)

    energy = calculate_total_energy(path_states)
    time_taken = steps * T_DRIVE

    # Console visualization
    visualize_path(high_res, smoothed_path, energy, steps, turns, time_taken)

    # Graphical visualization
    visualize_path_with_matplotlib(high_res, smoothed_path, energy, steps, turns, time_taken)


if __name__ == "__main__":
    main()
