import random
import matplotlib.pyplot as plt


def calculate_fitness(chromosome):
    x = int(''.join(map(str, chromosome)), 2)
    return x / 2


def create_population(population_size, chromosome_size):
    population = [[random.randint(0, 1) for _ in range(chromosome_size)] for _ in range(population_size)]
    return population


def select_parents(population):
    parents = random.sample(population, 2)
    return parents


def crossover(parents):
    cutoff_point = random.randint(1, len(parents[0]) - 1)
    child1 = parents[0][:cutoff_point] + parents[1][cutoff_point:]
    child2 = parents[1][:cutoff_point] + parents[0][cutoff_point:]
    return child1, child2


def mutate(individual, mutation_probability, fixed_mutation=False):
    if fixed_mutation:
        gene = random.randint(0, len(individual) - 1)
        individual[gene] = 1 - individual[gene]
    else:
        for i in range(len(individual)):
            if random.random() < mutation_probability:
                individual[i] = 1 - individual[i]
    return individual


def average_fitness(population):
    fitness_values = [calculate_fitness(individual) for individual in population]
    return sum(fitness_values) / len(fitness_values)


def best_individual(population):
    fitness_values = [calculate_fitness(individual) for individual in population]
    index_best = fitness_values.index(max(fitness_values))
    return population[index_best]


def plot_graph(average_fitnesses, best_fitnesses):
    generations = range(len(average_fitnesses))
    plt.plot(generations, average_fitnesses, label='Average Fitness')
    plt.plot(generations, best_fitnesses, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def display_population(population):
    for i, individual in enumerate(population):
        print(f'Individual {i + 1}: {individual}')


def genetic_algorithm(
        population_size,
        chromosome_size,
        crossover_probability,
        mutation_probability,
        fixed_mutation,
        num_generations
):
    population = create_population(population_size, chromosome_size)
    average_fitnesses = []
    best_fitnesses = []

    for generation in range(num_generations):
        new_population = []

        for _ in range(population_size // 2):
            parents = select_parents(population)
            child1, child2 = crossover(parents)

            child1 = mutate(child1, mutation_probability, fixed_mutation)
            child2 = mutate(child2, mutation_probability, fixed_mutation)

            new_population.extend([child1, child2])

        population = new_population

        current_average_fitness = average_fitness(population)
        current_best_fitness = calculate_fitness(best_individual(population))

        average_fitnesses.append(current_average_fitness)
        best_fitnesses.append(current_best_fitness)

    plot_graph(average_fitnesses, best_fitnesses)
    display_population(population)


# Input parameters
population_size = 10
chromosome_size = 20
crossover_probability = 0.8
mutation_probability = 0.1
fixed_mutation = True
num_generations = 50

# Run the genetic algorithm
genetic_algorithm(
    population_size,
    chromosome_size,
    crossover_probability,
    mutation_probability,
    fixed_mutation,
    num_generations
)
