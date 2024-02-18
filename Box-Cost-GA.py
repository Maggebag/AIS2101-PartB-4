import random
import matplotlib.pyplot as plt

# Define problem parameters
population_size = 10
max_generations = 400
mutation_rate = 0.01
crossover_rate = 0.75
min_dimension_value = 1
max_dimension_value = 100
dimension_costs = [15, 20, 10]

# Store parameters for printing
parameters = f"Population Size: {population_size}\nMutation Rate: {mutation_rate}\nCrossover Rate: {crossover_rate}"

# Function to generate a random set of XYZ values
def generate_xyz_in_range():
    dimensions = []
    for _ in range(3):
        dimensions.append(random.randint(min_dimension_value, max_dimension_value))
    return dimensions

# Function to generate an initial population
def generate_population():
    population = []
    for _ in range(population_size):
        population.append(generate_xyz_in_range())
    return population

# Calculate the cost of the box
def calculate_cost_pr_m3(dimensions):
    x_cost, y_cost, z_cost = dimension_costs
    x, y, z = dimensions
    volume = x * y * z

    # Define the cost of the different sides, scaled by the costs
    panel1 = (x * y) * 2 * x_cost
    panel2 = (x * z) * 2 * y_cost
    panel3 = (y * z) * 2 * z_cost
    cost = panel1 + panel2 + panel3  # Just the added cost of all sides of the box

    cost_pr_m3 = cost / volume  # Gives the total cost of the box pr m3
    return cost_pr_m3

# Select parents using Roulette wheel selection
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]

    # Spin that wheel
    selected_parents = random.choices(population, weights=probabilities, k=2)
    return selected_parents

# Perform uniform crossover
def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < crossover_rate:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

# Perform mutation
def mutate(individual):
    mutated_individual = [random.randint(min_dimension_value, max_dimension_value)
                          if random.random() < mutation_rate else individual[i] for i in range(len(individual))]
    return mutated_individual

# Perform genetic algorithm
def perform_genetic_algorithm(population):
    min_cost_pr_generation = []
    fitness_pr_generation = []
    lowest_cost_solution = None
    lowest_cost = float('inf')  # Initialize lowest cost to positive infinity

    for generation in range(max_generations):
        population_costs = [calculate_cost_pr_m3(individual) for individual in population]
        min_cost_pr_generation.append(min(population_costs))
        fitness_values = [1 / cost for cost in population_costs]
        fitness_pr_generation.append(min(fitness_values))

        print(f"Generation {generation + 1}, Cost: {min(population_costs)}, Fitness: {min(fitness_values)}")

        current_best_cost = min(population_costs)
        if current_best_cost < lowest_cost:
            lowest_cost = current_best_cost
            lowest_cost_solution = population[population_costs.index(current_best_cost)]

        # Start performing GA stuff
        new_population = []
        for _ in range(population_size):
            # Select parents
            parent1, parent2 = roulette_wheel_selection(population, fitness_values)

            # Perform crossover
            child1, child2 = uniform_crossover(parent1, parent2)

            # Mutate offspring
            child1 = mutate(child1)
            child2 = mutate(child2)

            # Add offspring to new population
            new_population.extend([child1, child2])

        # Replace old population with new population
        population = new_population

    # Plotting the minimum cost and fitness for each generation
    plt.plot(range(1, max_generations + 1), min_cost_pr_generation, label='Minimum Cost')
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.title(f'Cost Progress over Generations\nPopulation Size: {population_size}, Mutation Rate: {mutation_rate}, Crossover Rate: {crossover_rate}')
    plt.legend()

    lowest_cost_index = min_cost_pr_generation.index(lowest_cost) + 1  # Adding 1 because indices start from 0
    # Add red point at the location of lowest cost
    plt.scatter(lowest_cost_index, lowest_cost, color='red')
    plt.text(lowest_cost_index+5, lowest_cost-0.005, round(lowest_cost,4), fontsize=9)

    plt.show()

    print(f"Lowest Cost Found: {lowest_cost} at Generation {min_cost_pr_generation.index(lowest_cost) + 1}")
    print(f"Lowest Cost Solution Dimensions: {lowest_cost_solution}")


# Generate initial population
population = generate_population()

# Perform GA
perform_genetic_algorithm(population)
