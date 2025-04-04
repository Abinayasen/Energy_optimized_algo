import heapq
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import math
import random

# Constants
BATTERY_THRESHOLD = 20  # %
INITIAL_BATTERY = 100  # %
ENERGY_CAPACITY = 100000  # Joules
CARBON_INTENSITY = 0.8  # kg CO2 per kWh

P_DRIVE = 37.90  # W
P_CONTROL = 63.19  # W
P_LOAD = 54.71  # W

T_DRIVE = 33.70  # s
T_LOAD = 8.67  # s

class ChargingStation:
    def __init__(self, position, occupied=False):
        self.position = position
        self.occupied = occupied
        self.waiting_time = 0  # Time spent waiting at the station (calculated dynamically)

    def euclidean_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def calculate_waiting_time(self, num_queued_AGVs):
        """Calculate waiting time at a charging station based on queue size."""
        self.waiting_time = num_queued_AGVs * AVG_CHARGING_TIME  # Basic waiting time model
        return self.waiting_time

    def select_charging_station_ncs(self, agv_pos, stations):
        """Select the nearest charging station (NCS)."""
        min_distance = float('inf') #Finds closest charging station
        selected_station = None
        for station in stations:
            distance = self.euclidean_distance(agv_pos, station.position)
            if distance < min_distance and not station.occupied:
                min_distance = distance
                selected_station = station
        return selected_station
    def select_charging_station_mdcs(self, agv_pos, stations):
        """Select the charging station with the minimum delay (MDCS)."""
        min_delay = float('inf')
        selected_station = None
        for station in stations:
            distance = self.euclidean_distance(agv_pos, station.position)
            delay = station.calculate_waiting_time(len(stations))  # Assuming queue length
            total_delay = delay + distance
            if total_delay < min_delay and not station.occupied:
                min_delay = total_delay
                selected_station = station
        return selected_station


# ---- Grid Generation ----
def generate_grid(size, obstacle_percentage=0.1):
    grid = np.zeros((size, size))
    num_obstacles = int(size * size * obstacle_percentage)
    obstacles = random.sample([(x, y) for x in range(size) for y in range(size) if (x, y) != (0, 0) and (x, y) != (size - 1, size - 1)], num_obstacles)
    for obs in obstacles:
        grid[obs] = 1
    return grid

# ---- Energy-Optimized A* ----
def astar_energy_optimized(grid, start, goal, battery_level):
    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) * 5.2

    def neighbors(node):
        x, y = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                yield (nx, ny), dx, dy

    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, (0, 0), battery_level))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current_g, current, last_dir, battery_level = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], battery_level

        for neighbor, dx, dy in neighbors(current):
            move_cost = P_DRIVE
            if (dx, dy) != last_dir:
                move_cost += P_CONTROL
            if dx != last_dir[0] or dy != last_dir[1]:
                move_cost += 0.2 * P_DRIVE

            tentative_g = current_g + move_cost
            battery_used = move_cost * T_DRIVE
            remaining_battery = battery_level - (battery_used / ENERGY_CAPACITY * 100)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g, neighbor, (dx, dy), remaining_battery))

    return [], battery_level

# ---- Traditional A* ----
def astar_traditional(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        x, y = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                yield (nx, ny)

    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current_g, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g = current_g + P_DRIVE * T_DRIVE
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g, neighbor))
    return []

# ---- Energy Calculation ----
def calculate_energy(path):
    num_steps = len(path) - 1
    driving_energy = num_steps * P_DRIVE * T_DRIVE
    control_energy = num_steps * P_CONTROL * T_DRIVE
    load_handling_energy = num_steps * P_LOAD * T_LOAD
    return driving_energy + control_energy + load_handling_energy

def calculate_env_impact(energy_saved):
    energy_saved_kWh = energy_saved / 3600
    return energy_saved_kWh * CARBON_INTENSITY

# ---- Smooth Path ----
def smooth_path(path):
    if len(path) < 3:
        return path
    x = [p[1] for p in path]
    y = [p[0] for p in path]
    spline_x = CubicSpline(range(len(x)), x, bc_type='natural')
    spline_y = CubicSpline(range(len(y)), y, bc_type='natural')
    smoothed_x = spline_x(np.linspace(0, len(x) - 1, 100))
    smoothed_y = spline_y(np.linspace(0, len(y) - 1, 100))
    return list(zip(smoothed_y, smoothed_x))

# ---- Visualization with Charging Stations ----
def visualize_path_with_charging(grid, path, smoothed_path, title, stations=None, selected_station=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', color='red', markersize=5, label='A* Path')

    if smoothed_path:
        smooth_x, smooth_y = zip(*smoothed_path)
        plt.plot(smooth_y, smooth_x, color='blue', linewidth=2, label='Smoothed Path')

    if stations:
        for station in stations:
            x, y = station.position
            plt.scatter(y, x, c='green', s=200, label='Charging Station' if 'Charging Station' not in plt.gca().get_legend_handles_labels()[1] else "")
            if station.occupied:
                plt.text(y + 0.3, x + 0.3, f'{station.occupant_battery}%', fontsize=9)

    if selected_station:
        x_sel, y_sel = selected_station.position
        plt.scatter(y_sel, x_sel, facecolors='none', edgecolors='red', s=500, linewidths=2, label='Selected Station')

    agv_x, agv_y = path[0]
    plt.scatter(agv_y, agv_x, c='blue', marker='*', s=300, label='AGV Start')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, grid.shape[1])
    plt.ylim(-1, grid.shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
# AGV Class
class AGV:
    def __init__(self, position, battery_level=100):
        self.position = position
        self.battery_level = battery_level
        self.is_charging = False

    def move_to_station(self, station):
        """Simulate moving AGV to a charging station."""
        print(f"AGV moving to charging station at position {station.position}.")
        self.position = station.position

    def charge(self, target_soc, charge_time):
        """Simulate charging process."""
        print(f"Charging AGV to {target_soc}% SoC.")
        self.is_charging = True
        time.sleep(charge_time / 60)  # Simulate time taken for charging
        self.battery_level = target_soc
        self.is_charging = False

    def needs_charging(self):
        """Check if AGV needs charging based on its current battery level."""
        return self.battery_level < BATTERY_THRESHOLD

# Function to calculate charging time
def calculate_charging_time(dod, target_soc):
    """Calculate the charging time based on DoD and target SoC."""
    if target_soc == 100:
        return CHARGING_DURATIONS[100] * math.exp(0.3746 * dod)  # For 100% SoC
    elif target_soc == 95:
        return CHARGING_DURATIONS[95] * math.exp(0.3706 * dod)  # For 95% SoC
    elif target_soc == 90:
        return CHARGING_DURATIONS[90] * math.exp(0.3746 * dod)  # For 90% SoC
    else:
        return 0  # Invalid SoC

import time

# Constants (As defined earlier)
BATTERY_THRESHOLD = 20  # %
INITIAL_BATTERY = 100  # %
ENERGY_CAPACITY = 100000  # Joules
CARBON_INTENSITY = 0.8  # kg CO2 per kWh

P_DRIVE = 37.90  # W
P_CONTROL = 63.19  # W
P_LOAD = 54.71  # W

T_DRIVE = 33.70  # s
T_LOAD = 8.67  # s

AVG_CHARGING_TIME = 15  # Minutes (adjustable, example value)

CHARGING_DURATIONS = {100: 31.365, 95: 19.055, 90: 11.809}  # Sample charging duration per SoC


def main():
    # Step 1: Create a grid (for pathfinding)
    grid_size = 60  # Increased grid size for larger paths
    grid = generate_grid(grid_size, obstacle_percentage=0.15)

    # Step 2: Create Charging Stations (example positions)
    station_positions = [(0, 0), (5, 5), (10, 10), (15, 15)]
    stations = [ChargingStation(position=pos) for pos in station_positions]

    # Step 3: Create AGVs (AGVs will have random initial positions)
    agv_positions = [(1, 2), (3, 4), (6, 8), (15, 16), (18, 19)]
    agvs = [AGV(position=pos, battery_level=INITIAL_BATTERY) for pos in agv_positions]

    # Step 4: Iterate over AGVs to simulate their charging and movement behavior
    for agv in agvs:
        print(f"Initial state of AGV at {agv.position} with battery level: {agv.battery_level}%")

        # Step 5: Check if AGV needs charging before pathfinding
        selected_station = None  # Ensure selected_station is initialized
        if agv.needs_charging():
            print(f"AGV at position {agv.position} needs charging.")

            # Select a charging station
            selected_station = stations[0].select_charging_station_ncs(agv.position, stations)
            print("Using Nearest Charging Station (NCS) strategy.")

            # AGV moves to the selected charging station
            agv.move_to_station(selected_station)

            # Choose a target SoC for charging (e.g., 95%)
            target_soc = 95  # Example target SoC
            dod = 100 - agv.battery_level  # Depth of Discharge (DoD)

            # Calculate charging time
            charge_time = calculate_charging_time(dod, target_soc)

            # AGV charges to the selected SoC
            agv.charge(target_soc, charge_time)
            print(f"AGV charged to {target_soc}% SoC.\n")

        else:
            print(f"AGV at position {agv.position} does not need charging.\n")

        # Simulate pathfinding for AGV (energy-optimized or traditional A*)
        start_position = agv.position
        goal_position = (grid_size - 1, grid_size - 1)  # Example goal position (bottom-right corner)

        # Energy-Optimized Pathfinding
        print(f"Calculating energy-optimized path for AGV at {start_position} to goal {goal_position}...")
        path, remaining_battery = astar_energy_optimized(grid, start_position, goal_position, agv.battery_level)

        if path:
            print(f"Energy-optimized path found: {path}")
            print(f"Remaining battery after pathfinding: {remaining_battery}%")
            energy_used = calculate_energy(path)
            print(f"Energy used for the path: {energy_used} Joules")
            environmental_impact = calculate_env_impact(energy_used)
            print(f"Environmental impact (carbon emissions): {environmental_impact} kg CO2\n")

            # Check if battery level is below threshold after pathfinding
            if remaining_battery < BATTERY_THRESHOLD:
                print(f"AGV's battery level is now {remaining_battery}%. Moving to charging station.")
                # Select a charging station for AGV to go to after pathfinding
                selected_station = stations[0].select_charging_station_ncs(agv.position, stations)
                agv.move_to_station(selected_station)
                charge_time = calculate_charging_time(100 - agv.battery_level, 100)  # Charge to 100%
                agv.charge(100, charge_time)
                print(f"AGV charged to 100% SoC after pathfinding.\n")

            # Visualize the energy-optimized path
            smoothed_path = smooth_path(path)
            visualize_path_with_charging(grid, path, smoothed_path, "Energy-Optimized Path with Charging Stations", stations, selected_station)

        else:
            print("No path found using energy-optimized A*.\n")

        # Traditional A* Pathfinding (as a fallback)
        print(f"Calculating traditional path for AGV at {start_position} to goal {goal_position}...")
        path_traditional = astar_traditional(grid, start_position, goal_position)

        if path_traditional:
            print(f"Traditional path found: {path_traditional}")
            energy_used_traditional = calculate_energy(path_traditional)
            print(f"Energy used for traditional path: {energy_used_traditional} Joules")
            environmental_impact_traditional = calculate_env_impact(energy_used_traditional)
            print(f"Environmental impact (carbon emissions): {environmental_impact_traditional} kg CO2\n")

            # Visualize the traditional path
            smoothed_path_traditional = smooth_path(path_traditional)
            visualize_path_with_charging(grid, path_traditional, smoothed_path_traditional, "Traditional Path with Charging Stations", stations)

        else:
            print("No path found using traditional A*.\n")


# Execute the main function
if __name__ == "__main__":
    main()

