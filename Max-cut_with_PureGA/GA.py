import random
import time

# 유전 알고리즘 매개변수
POPULATION_SIZE = 100
MUTATION_RATE = 0.05  # 돌연변이 확률
NUM_GENERATIONS = 100
ELITISM_COUNT = 2  # 각 세대에서 보존할 엘리트 개체의 수

def read_graph(filename):
    graph = {}
    with open(filename, 'r') as file:
        num_vertices, num_edges = map(int, file.readline().split())
        for _ in range(num_edges):
            u, v, weight = map(int, file.readline().split())
            if u not in graph:
                graph[u] = {}
            if v not in graph:
                graph[v] = {}
            graph[u][v] = weight
            graph[v][u] = weight
    return graph

def evaluate_cut(graph, cut):
    cut_weight = 0
    for u in cut:
        for v, weight in graph[u].items():
            if v not in cut:
                cut_weight += weight
    return cut_weight

def generate_initial_population(graph, population_size):
    population = []
    vertices = list(graph.keys())
    for _ in range(population_size):
        # 개체를 더 다양하게 생성하기 위해, 반을 임의로 선택하는 대신 전체 정점 집합에서 임의로 선택
        random_cut = random.sample(vertices, k=random.randint(1, len(vertices)-1))
        population.append(random_cut)
    return population

def crossover(parent1, parent2):
    if min(len(parent1), len(parent2)) <= 2:
        return parent1.copy(), parent2.copy()
    else:
        cut_point = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child1 = parent1[:cut_point] + [gene for gene in parent2 if gene not in parent1[:cut_point]]
        child2 = parent2[:cut_point] + [gene for gene in parent1 if gene not in parent2[:cut_point]]
        return child1, child2

def mutate(individual):
    # 돌연변이를 일으킬 때, 단순히 위치를 교환하는 것이 아니라, 새로운 정점을 추가하거나 제거하는 방식을 추가
    mutation_type = random.random()
    if mutation_type < MUTATION_RATE:
        if random.random() > 0.5 and len(individual) > 1:
            # 정점 제거
            individual.remove(random.choice(individual))
        else:
            # 새로운 정점 추가
            possible_additions = [v for v in graph.keys() if v not in individual]
            if possible_additions:
                individual.append(random.choice(possible_additions))

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected[0], selected[1]

def genetic_algorithm(graph):
    best_cut_weight = float('-inf')
    best_individual = None

    population = generate_initial_population(graph, POPULATION_SIZE)
    for generation in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_cut(graph, individual) for individual in population]

        # 엘리티즘: 최고의 개체를 보존
        elites_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ELITISM_COUNT]
        elites = [population[i] for i in elites_indices]

        new_population = elites.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = roulette_wheel_selection(population, fitness_scores), roulette_wheel_selection(population, fitness_scores)
            # 튜플 대신 개별 개체로 반환된 부모를 처리
            child1, child2 = crossover(parent1[0], parent2[0])
            mutate(child1)
            mutate(child2)
            new_population += [child1, child2]

        population = new_population[:POPULATION_SIZE]

        # 최고 해 업데이트
        current_max_fitness = max(fitness_scores)
        if current_max_fitness > best_cut_weight:
            best_cut_weight = current_max_fitness
            best_individual = population[fitness_scores.index(current_max_fitness)]

    return best_individual, best_cut_weight


def write_solution_to_file(solution, filename="maxcut.out"):
    with open(filename, 'w') as file:
        file.write(" ".join(map(str, solution)))

if __name__ == "__main__":
    graph = read_graph("maxcut.in")
    best_individual, best_cut_weight = genetic_algorithm(graph)
    write_solution_to_file(best_individual)
    #print("Total weight of the selected cut:", best_cut_weight)