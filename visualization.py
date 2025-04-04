import pygame
import numpy as np
import heapq
import random
import time

# === Constants ===
GRID_SIZE = 45
CELL_SIZE = 16
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
GREEN = (50, 205, 50)
RED = (255, 0, 0)
BLUE = (30, 144, 255)
YELLOW = (255, 255, 0)

START = (0, 0)
GOAL = (44, 44)
CHARGING_STATIONS = [(0, 39), (20, 20), (39, 0)]

INITIAL_BATTERY = 100
ENERGY_CAPACITY = 100000  # in Joules

# Energy constants
P_DRIVE = 37.9
P_CONTROL = 63.19
T_DRIVE = 33.7

# === Pathfinding Functions ===

def generate_grid(size, obstacle_percentage=0.10):
    grid = np.zeros((size, size))
    num_obstacles = int(size * size * obstacle_percentage)
    for _ in range(num_obstacles):
        while True:
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            if (x, y) not in [START, GOAL] + CHARGING_STATIONS and grid[x][y] == 0:
                grid[x][y] = 1
                break
    return grid

def is_valid_diagonal(grid, x, y, dx, dy):
    if dx != 0 and dy != 0:
        if grid[x][y + dy] == 1 or grid[x + dx][y] == 1:
            return False
    return True

def heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) * 5.2

def astar_energy_optimized(grid, start, goal, battery_level):
    def find_nearest_station(pos):
        return min(CHARGING_STATIONS, key=lambda c: heuristic(pos, c))

    target = find_nearest_station(start) if battery_level < 20 else goal

    def neighbors(node):
        x, y = node
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx][ny] == 0 and is_valid_diagonal(grid, x, y, dx, dy):
                    yield (nx, ny), dx, dy

    open_list = []
    heapq.heappush(open_list, (heuristic(start, target), 0, start, (0, 0), battery_level))
    came_from = {}
    g_score = {start: 0}

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
            if (dx, dy) != last_dir:
                move_cost += P_CONTROL
            if dx != last_dir[0] or dy != last_dir[1]:
                move_cost += 0.2 * P_DRIVE

            tentative_g = current_g + move_cost
            battery_used = move_cost * T_DRIVE
            remaining_battery = battery_level - (battery_used / ENERGY_CAPACITY * 100)

            if remaining_battery <= 0:
                continue

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_list, (tentative_g + heuristic(neighbor, target), tentative_g, neighbor, (dx, dy), remaining_battery))

    return [], battery_level

def astar_traditional(grid, start, goal):
    def neighbors(node):
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny] == 0:
                yield (nx, ny)

    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    came_from = {}
    g_score = {start: 0}

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
                heapq.heappush(open_list, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))

    return []

def calculate_energy(path):
    steps = len(path) - 1
    return steps * P_DRIVE * T_DRIVE

# === Pygame Setup ===
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 60))
pygame.display.set_caption("A* Pathfinding Visualization")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

grid = generate_grid(GRID_SIZE)
path_traditional = astar_traditional(grid, START, GOAL)
energy_traditional = calculate_energy(path_traditional)

def draw_grid():
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = BLACK if grid[x][y] == 1 else GREY
            pygame.draw.rect(screen, color, rect, 0 if grid[x][y] else 1)

def draw_agent(pos, color):
    pygame.draw.rect(screen, color, (pos[1]*CELL_SIZE, pos[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_stats(path_type, battery, energy):
    pygame.draw.rect(screen, WHITE, (0, SCREEN_SIZE, SCREEN_SIZE, 60))
    texts = [
        f"Path Type: {path_type}",
        f"Battery: {battery:.1f}%",
        f"Energy: {energy:.1f} J"
    ]
    for i, text in enumerate(texts):
        screen.blit(font.render(text, True, BLACK), (10, SCREEN_SIZE + 5 + 20 * i))

def animate_path(path, is_optimized):
    battery = INITIAL_BATTERY
    energy_used = 0
    for i, pos in enumerate(path):
        if battery <= 0:
            msg = font.render("Battery Depleted. Agent stopped.", True, RED)
            screen.blit(msg, (10, SCREEN_SIZE + 40))
            pygame.display.flip()
            pygame.time.wait(3000)
            return

        screen.fill(WHITE)
        draw_grid()

        for cs in CHARGING_STATIONS:
            pygame.draw.rect(screen, YELLOW, (cs[1]*CELL_SIZE, cs[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.draw.rect(screen, RED, (START[1]*CELL_SIZE, START[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, BLUE, (GOAL[1]*CELL_SIZE, GOAL[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for p in path[:i+1]:
            draw_agent(p, GREEN if is_optimized else RED)

        if i > 0:
            energy_used += P_DRIVE * T_DRIVE
            battery -= (P_DRIVE * T_DRIVE) / ENERGY_CAPACITY * 100

        draw_stats("Optimized" if is_optimized else "Traditional", battery, energy_used)

        pygame.display.flip()
        time.sleep(0.05)
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    msg = font.render("                                                                           Path complete. Press [C] to continue.", True, BLACK)
    screen.blit(msg, (10, SCREEN_SIZE + 40))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    return

def main():
    running = True
    while running:
        screen.fill(WHITE)
        draw_grid()

        for cs in CHARGING_STATIONS:
            pygame.draw.rect(screen, YELLOW, (cs[1]*CELL_SIZE, cs[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.draw.rect(screen, RED, (START[1]*CELL_SIZE, START[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, BLUE, (GOAL[1]*CELL_SIZE, GOAL[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        msg = font.render("Press [O] Optimized | [T] Traditional | [Q] Quit", True, BLACK)
        screen.blit(msg, (10, SCREEN_SIZE + 20))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_o:
                    path_energy, battery_remaining = astar_energy_optimized(grid, START, GOAL, INITIAL_BATTERY)
                    if path_energy:
                        print(f"Optimized path found: {len(path_energy)} steps.")
                        animate_path(path_energy, is_optimized=True)
                    else:
                        print("No optimized path found.")
                elif event.key == pygame.K_t:
                    animate_path(path_traditional, is_optimized=False)

    pygame.quit()

main()
