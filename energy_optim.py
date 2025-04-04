#Battery < 0 == stop vehicle.
#Instead of battery lesser than threshold function, calculate the distance from all battery stations, make sure battery doesnt run out. Theliva yosichitu poderen. Olaruren.

import heapq
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import time
import random

BATTERY_THRESHOLD = 20  # in percentage
INITIAL_BATTERY = 100  # 100% battery
ENERGY_CAPACITY = 100000  # Total energy capacity in Joules
charging_station = (0, 39)  # Example fixed charging station position


# Energy Constants
P_DRIVE = 37.90  # W
P_CONTROL = 63.19  # W
P_LOAD = 54.71  # W

T_DRIVE = 33.70  # s
T_LOAD = 8.67  # s

# Carbon intensity in kg CO2 per kWh (Average India Grid)
CARBON_INTENSITY = 0.8  # kg CO2 per kWh (India's average grid intensity)

# Energy-Optimized A* Algorithm (Using Smooth A*)
def astar_energy_optimized(grid, start, goal, battery_level):
    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) * 5.2  # Diagonal heuristic for 8 directions

    def neighbors(node):
        x, y = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                yield (nx, ny), dx, dy

    # If battery < threshold, redirect to charging station
    target = charging_station if battery_level < BATTERY_THRESHOLD else goal

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, target), 0, start, (0, 0), battery_level))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}

    while open_list:
        _, current_g, current, last_dir, battery_level = heapq.heappop(open_list)

        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], battery_level

        for neighbor, dx, dy in neighbors(current):
            move_cost = P_DRIVE
            # Penalize unnecessary turns (Energy for control)
            if (dx, dy) != last_dir:
                move_cost += P_CONTROL
            # Acceleration penalty for sharp direction change
            if dx != last_dir[0] or dy != last_dir[1]:
                move_cost += 0.2 * P_DRIVE  # Simulate energy loss

            tentative_g = current_g + move_cost
            battery_used = move_cost * T_DRIVE
            remaining_battery = battery_level - (battery_used / ENERGY_CAPACITY * 100)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, target)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g, neighbor, (dx, dy), remaining_battery))

    return [], battery_level

# Traditional A* Algorithm (Basic A*)
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
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
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

# Path Smoothing (for both paths)
def smooth_path(path):
    if len(path) < 3:
        return path

    x = [p[1] for p in path]
    y = [p[0] for p in path]

    spline_x = CubicSpline(range(len(x)), x, bc_type='natural')
    spline_y = CubicSpline(range(len(y)), y, bc_type='natural')

    smoothed_x = spline_x(np.linspace(0, len(x) - 1, 50))
    smoothed_y = spline_y(np.linspace(0, len(y) - 1, 50))

    return list(zip(smoothed_y, smoothed_x))

# Generate a 40x40 Grid with Random Obstacles
def generate_grid(size, obstacle_percentage=0.3):
    grid = np.zeros((size, size))

    # Place random obstacles
    num_obstacles = int(size * size * obstacle_percentage)
    obstacles = random.sample([(x, y) for x in range(size) for y in range(size) if (x, y) != (0, 0) and (x, y) != (size - 1, size - 1)], num_obstacles)

    for obs in obstacles:
        grid[obs] = 1

    return grid

# Calculate Total Energy Consumption
def calculate_energy(path, is_optimized=False):
    num_steps = len(path) - 1  # Subtracting one since the first point doesn't involve moving

    # Calculate driving energy and control energy
    driving_energy = num_steps * P_DRIVE * T_DRIVE
    control_energy = num_steps * P_CONTROL * T_DRIVE

    # Load handling energy (only considered if handling required on each step)
    load_handling_energy = num_steps * P_LOAD * T_LOAD

    total_energy = driving_energy + control_energy + load_handling_energy

    return total_energy

# Calculate Environmental Impact (CO2 Savings)
def calculate_env_impact(energy_saved):
    energy_saved_kWh = energy_saved / 3600  # Convert Joules to kWh
    co2_saved = energy_saved_kWh * CARBON_INTENSITY  # kg CO2
    return co2_saved

# Visualize Paths
def visualize_path(grid, path, smoothed_path, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', color='red', markersize=5, label='A* Path')

    if smoothed_path:
        smooth_x, smooth_y = zip(*smoothed_path)
        plt.plot(smooth_y, smooth_x, marker='o', color='green', markersize=5, label='Smoothed Path')

    plt.legend()
    plt.title(title)
    plt.show()

# Main Execution
size = 40  # Grid size 40x40
start = (0, 0)
goal = (39, 39)

# Generate a 40x40 grid with 30% obstacles
grid = generate_grid(size, obstacle_percentage=0.3)

# Run Energy-Optimized A* and Traditional A*
path_energy, battery_remaining = astar_energy_optimized(grid, start, goal, INITIAL_BATTERY)
path_traditional = astar_traditional(grid, start, goal)

# Smooth the Paths
smoothed_path_energy = smooth_path(path_energy)
smoothed_path_traditional = smooth_path(path_traditional)

# Energy Calculation for Both Paths
energy_optimized = calculate_energy(path_energy, is_optimized=True)
energy_traditional = calculate_energy(path_traditional)

# Energy Savings Calculation
energy_savings = energy_traditional - energy_optimized

# Energy Efficiency Ratio (EER)
eer = energy_traditional / energy_optimized

# Percentage Energy Savings
percentage_savings = (energy_traditional - energy_optimized) / energy_traditional * 100

# Environmental Impact (CO2 Savings)
co2_saved = calculate_env_impact(energy_savings)

# Visualization
visualize_path(grid, path_energy, smoothed_path_energy, "Energy-Optimized A* Path")
visualize_path(grid, path_traditional, smoothed_path_traditional, "Traditional A* Path")

# Print Results
print(f"Energy Consumed by Optimized A*: {energy_optimized:.2f} Joules")
print(f"Energy Consumed by Traditional A*: {energy_traditional:.2f} Joules")
print(f"Energy Savings by Using Optimized A*: {energy_savings:.2f} Joules")
print(f"Energy Efficiency Ratio (EER): {eer:.2f}")
print(f"Percentage Energy Savings: {percentage_savings:.2f}%")
print(f"Environmental Impact: {co2_saved:.2f} kg CO2 saved")

# Plotting the energy comparison graph
energy_values = [energy_traditional, energy_optimized]
algorithms = ["Traditional A*", "Optimized A*"]

plt.bar(algorithms, energy_values, color=['red', 'green'])
plt.title('Energy Consumption Comparison (40x40 Grid with Scattered Obstacles)')
plt.xlabel('Algorithm')
plt.ylabel('Energy Consumed (Joules)')
plt.show()
