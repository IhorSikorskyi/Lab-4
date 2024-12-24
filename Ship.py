import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Завантаження і очищення даних
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df_new = df.iloc[:, [2, 4]]  # Стовпці з координатами
    df_clean = preper(df_new)
    return df_clean

def preper(df):
    print("Кількість пропущених значень у кожному стовпці:")
    print(df.isnull().sum())
    df_clean = df.dropna(axis=1)
    print("\nПісля очищення:")
    print(df_clean.isnull().sum())
    return df_clean

# Генерація координат
def generate_random_coords(df_clean, num_points=15):
    unique_values = df_clean.iloc[:, 1].unique()[:num_points]  # Вибираємо перші 15 точок
    random_coords = {value: (np.random.uniform(-100, 100), np.random.uniform(-100, 100)) for value in unique_values}
    return np.array(list(random_coords.values()))  # Список координат

# Створення матриці відстаней
def create_distance_matrix(coords):
    return cdist(coords, coords, metric='euclidean')

# Функція пристосованості: обчислення загальної довжини маршруту
def fitness(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    return total_distance

# Ініціалізація параметрів PSO
def initialize_particles(n_particles, coords):
    particles = [np.random.permutation(len(coords)) for _ in range(n_particles)]
    velocities = [np.random.uniform(-1, 1, len(coords)) for _ in range(n_particles)]
    return particles, velocities

# Оновлення найкращих рішень
def update_best_solutions(particles, p_best, p_best_scores, g_best, g_best_score, distance_matrix):
    for i, particle in enumerate(particles):
        current_fitness = fitness(particle, distance_matrix)
        if current_fitness < p_best_scores[i]:
            p_best[i] = particle
            p_best_scores[i] = current_fitness
        if current_fitness < g_best_score:
            g_best = particle
            g_best_score = current_fitness
    return p_best, p_best_scores, g_best, g_best_score

# Основний цикл PSO
def run_pso(n_iterations, particles, velocities, p_best, p_best_scores, g_best, g_best_score, distance_matrix):
    history = []
    for iteration in range(n_iterations):
        for i in range(len(particles)):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = 0.5 * velocities[i] + 1.5 * r1 * (p_best[i] - particles[i]) + 1.5 * r2 * (g_best - particles[i])
            particles[i] = np.argsort(velocities[i])  # Оновлення позиції частинок
        p_best, p_best_scores, g_best, g_best_score = update_best_solutions(particles, p_best, p_best_scores, g_best, g_best_score, distance_matrix)
        history.append(g_best_score)
    return g_best, g_best_score, history

# Візуалізація маршрутів
def visualize_sample_routes(coords, sample_size=100):
    n = len(coords)
    sampled_routes = [np.random.permutation(n) for _ in range(sample_size)]
    plt.figure(figsize=(12, 8))
    for route in sampled_routes:
        route_coords = coords[list(route)]
        route_coords = np.vstack([route_coords, route_coords[0]])  # Замкнути маршрут
        plt.plot(route_coords[:, 0], route_coords[:, 1], 'gray', alpha=0.3)
    plt.scatter(coords[:, 0], coords[:, 1], c='red', label='Точки')
    plt.title("Приклади можливих маршрутів")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plot_best_route(coords, g_best):
    plt.figure(figsize=(12, 8))
    route_coords = coords[list(g_best)]
    route_coords = np.vstack([route_coords, route_coords[0]])  # Замкнути маршрут
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-', label="Найкращий маршрут")
    plt.scatter(coords[:, 0], coords[:, 1], c='red', label='Точки')
    plt.title("Найкращий знайдений маршрут")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Мінімальна вартість')
    plt.title("Прогрес алгоритму PSO")
    plt.xlabel("Ітерація")
    plt.ylabel("Вартість")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    file_path = 'Container Transporty.csv'
    df_clean = load_and_clean_data(file_path)
    coords = generate_random_coords(df_clean)
    distance_matrix = create_distance_matrix(coords)

    n_particles = 30
    n_iterations = 100
    particles, velocities = initialize_particles(n_particles, coords)

    # Початкове знаходження найкращих рішень
    p_best = particles.copy()
    p_best_scores = [fitness(p, distance_matrix) for p in particles]
    g_best = p_best[np.argmin(p_best_scores)]
    g_best_score = min(p_best_scores)

    # Запуск PSO
    g_best, g_best_score, history = run_pso(n_iterations, particles, velocities, p_best, p_best_scores, g_best, g_best_score, distance_matrix)

    # Виведення результатів
    print(f"Найкращий знайдений маршрут: {g_best}")
    print(f"Мінімальна відстань: {g_best_score}")

    # Візуалізація результатів
    visualize_sample_routes(coords)
    plot_best_route(coords, g_best)
    plot_history(history)
