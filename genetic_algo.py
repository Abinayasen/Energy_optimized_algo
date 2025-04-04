import random
import matplotlib.pyplot as plt
import numpy as np

# Constants
GRID_WIDTH = 10
GRID_HEIGHT = 10
START = (0, 0)
GOAL = (9, 9)
MUTATION_RATE = 0.1 #change of mutation in offspring
POPULATION_SIZE = 100 #paths per generation
GENERATIONS = 500 #algorithm runs 500 iterations

# Directions: Up, Down, Left, Right
DIRECTIONS =  [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Create grid with some obstacles
def create_grid():
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    # Adding random obstacles
    for _ in range(20):  # 20 obstacles
        x = random.randint(0, GRID_HEIGHT - 1)
        y = random.randint(0, GRID_WIDTH - 1)
        grid[x][y] = 1  # 1 represents an obstacle
    grid[START[0]][START[1]] = 0  # Start point is open
    grid[GOAL[0]][GOAL[1]] = 0  # Goal point is open
    return grid

# Fitness function: Path length + penalty for collisions with obstacles
def fitness(path, grid):
    if path[-1] != GOAL:
        return float('inf')
    length = len(path)
    penalty = 0
    for (x, y) in path:
        if grid[x][y] == 1:  # If there's an obstacle
            penalty += 10
    return length + penalty

# Generate a random path from start to goal
def random_path():
    path = [START]
    while path[-1] != GOAL:
        current = path[-1]
        next_move = random.choice(DIRECTIONS)
        new_position = (current[0] + next_move[0], current[1] + next_move[1])
        # Ensure the move is within bounds and not blocked by obstacles
        if 0 <= new_position[0] < GRID_HEIGHT and 0 <= new_position[1] < GRID_WIDTH:
            path.append(new_position)
        # Stop if we're stuck (for simplicity, we just stop here)
        if len(path) > 100:
            break
    return path

# Selection: Tournament selection
def select(population, grid):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda path: fitness(path, grid))
    return tournament[0]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation: Randomly mutate the path
def mutate(path, grid):
    if random.random() < MUTATION_RATE:
        # Choose a random index to mutate
        idx = random.randint(1, len(path) - 1)
        current = path[idx]
        next_move = random.choice(DIRECTIONS)
        new_position = (current[0] + next_move[0], current[1] + next_move[1])
        # Ensure the mutation is within bounds and not blocked
        if 0 <= new_position[0] < GRID_HEIGHT and 0 <= new_position[1] < GRID_WIDTH:
            path[idx] = new_position
    return path

# Visualization function
def plot_grid(grid, population, best_path=None):
    plt.figure(figsize=(6, 6))

    # Display the grid
    plt.imshow(grid, cmap="Greys", origin="upper", extent=[0, GRID_WIDTH, 0, GRID_HEIGHT])

    # Highlight the start and goal
    plt.scatter(START[1], START[0], color='green', s=100, label='Start')
    plt.scatter(GOAL[1], GOAL[0], color='red', s=100, label='Goal')

    # Plot all paths in the population
    for path in population:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', alpha=0.3, linewidth=1)

    # Highlight the best path found
    if best_path:
        best_x, best_y = zip(*best_path)
        plt.plot(best_y, best_x, color='yellow', linewidth=2, label='Best Path')

    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y axis to match grid coordinates
    plt.title('Pathfinding using Genetic Algorithm')
    plt.show()
# Calculate number of turns and distance for the path
def calculate_turns_and_distance(path):
    distance = len(path) - 1  # Distance is the number of steps
    turns = 0
    for i in range(1, len(path) - 1):
        prev_direction = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_direction != next_direction:
            turns += 1
    print(f'Number of turns : {turns}, distance : {distance}')

# Main genetic algorithm
def genetic_algorithm():
    grid = create_grid()
    population = [random_path() for _ in range(POPULATION_SIZE)]
    for generation in range(GENERATIONS):
        population.sort(key=lambda path: fitness(path, grid))
        best_path = population[0]

        # Visualization step
        if generation % 10 == 0:  # Display every 10 generations
            plot_grid(grid, population, best_path)

        if fitness(best_path, grid) == len(best_path):
            print(f"Path found in {generation} generations!")
            print("Path:", best_path)
            plot_grid(grid, population, best_path)
            calculate_turns_and_distance(best_path)
            break

        # Generate the next generation
        next_generation = population[:10]  # Keep the best 10
        while len(next_generation) < POPULATION_SIZE:
            parent1 = select(population, grid)
            parent2 = select(population, grid)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1, grid), mutate(child2, grid)])

        population = next_generation

# Run the genetic algorithm with visualization
genetic_algorithm()
