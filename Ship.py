import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pyswarm import pso

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
    unique_values = df_clean.iloc[:, 1].unique()[:num_points]
    random_coords = {value: (np.random.uniform(-100, 100), np.random.uniform(-100, 100)) for value in unique_values}
    return np.array(list(random_coords.values()))

# Створення матриці відстаней
def create_distance_matrix(coords):
    return cdist(coords, coords, metric='euclidean')

# Функція пристосованості: обчислення загальної довжини маршруту
def fitness(route, distance_matrix):
    route = np.argsort(route)
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    total_distance += distance_matrix[route[-1], route[0]]  # Замикання маршруту
    return total_distance

# Основний алгоритм PSO з pyswarm
def optimize_with_pso(distance_matrix, num_particles=30, num_iterations=100):
    num_points = distance_matrix.shape[0]
    # Межі для частинок (всі значення в діапазоні [0, num_points])
    lb = [0] * num_points
    ub = [num_points - 1] * num_points

    history = []
    iteration_counter = [0]

    def fitness_with_history(route):
        cost = fitness(route, distance_matrix)
        if iteration_counter[0] % 10 == 0:
            history.append(cost)
        iteration_counter[0] += 1
        return cost

    # Виконання PSO
    best_route, best_distance = pso(
        fitness_with_history,
        lb, ub, swarmsize=num_particles, maxiter=num_iterations, debug=True
    )
    return np.argsort(best_route), best_distance, history

# Візуалізація маршрутів
def visualize_sample_routes(coords, sample_size=100):
    n = len(coords)
    sampled_routes = [np.random.permutation(n) for _ in range(sample_size)]
    plt.figure(figsize=(12, 8))
    for route in sampled_routes:
        route_coords = coords[list(route)]
        route_coords = np.vstack([route_coords, route_coords[0]])
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
    route_coords = np.vstack([route_coords, route_coords[0]])
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-', label="Найкращий маршрут")
    plt.scatter(coords[:, 0], coords[:, 1], c='red', label='Точки')
    plt.title("Найкращий знайдений маршрут")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Візуалізація історії
def plot_history(history):
    smoothed_history = pd.Series(history).rolling(window=5).mean()  # Згладжування (середнє по 5 сусіднім точкам)

    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(history) * 10, 10), smoothed_history, label='Згладжена мінімальна вартість', color='blue',
             linewidth=2)
    plt.scatter(range(0, len(history) * 10, 10), history, color='red', s=10, label='Вартість (кожна 10-а ітерація)',
                alpha=0.6)

    plt.title("Прогрес алгоритму PSO", fontsize=16)
    plt.xlabel("Ітерація", fontsize=14)
    plt.ylabel("Вартість", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'Container Transporty.csv'
    df_clean = load_and_clean_data(file_path)
    coords = generate_random_coords(df_clean)
    distance_matrix = create_distance_matrix(coords)

    # Оптимізація маршруту за допомогою PSO
    best_route, best_distance, history = optimize_with_pso(distance_matrix)

    # Виведення результатів
    print(f"Найкращий знайдений маршрут: {best_route}")
    print(f"Мінімальна відстань: {best_distance}")

    # Візуалізація результатів
    visualize_sample_routes(coords)
    plot_best_route(coords, best_route)
    plot_history(history)
