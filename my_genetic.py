import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time


# Test Function
def sphere(x):
    return sum(x ** 2)


# Problem to be defined here as a class
class Problem:

    def __init__(self):
        self.cost_function = sphere  # cost function
        self.min_gene_val = -10
        self.max_gene_val = 10
        self.num_genes = 5


class Individual:
    chromosome = None
    cost = None

    def __init__(self, problem=None):

        if problem is not None:
            self.chromosome = np.random.uniform(problem.min_gene_val,
                                                problem.max_gene_val,
                                                problem.num_genes)
            self.cost = problem.cost_function(self.chromosome)

    def crossover(self, other_individual, explore):
        child1 = deepcopy(self)
        child2 = deepcopy(other_individual)

        alpha = np.random.uniform(-explore, 1 + explore, child1.chromosome.shape)

        child1.chromosome = (alpha * self.chromosome) + ((1 - alpha) * other_individual.chromosome)
        child2.chromosome = (alpha * other_individual.chromosome) + ((1 - alpha) * self.chromosome)

        return child1, child2

    # Mutate method
    def mutate(self, mutation_probability, mutation_range):

        for i in range(self.chromosome.shape[0]):
            if np.random.rand() < mutation_probability:
                self.chromosome[i] += mutation_range * np.random.rand()


class TestGroup:
    # These class attributes define the set of values that will be used for
    # each Test that is executed. These settings can be over-ridden when
    # instantiating a TestGroup object. Default values are set here to prevent
    # someone instantiating a TestGroup object without the minimum
    def __init__(self,
                 group_sample_size=50,
                 group_population_sizes=(25, 50, 100),
                 group_max_iterations=(10, 25, 50),
                 group_explore_rates=(0.1, 0.2, 0.3),
                 group_mutation_probs=(0.05, 0.1, 0.2, 0.5),
                 group_mutation_ranges=(0.1, 0.2, 0.3),
                 ):
        self.group_sample_size = group_sample_size
        self.group_population_sizes = group_population_sizes
        self.group_max_iterations = group_max_iterations
        self.group_explore_rates = group_explore_rates
        self.group_mutation_probs = group_mutation_probs
        self.group_mutation_ranges = group_mutation_ranges
        self.group_results = []
        self.test_id = 0

    def group_run_tests(self):
        print("\n\n======================================\nRunning population value tests...")
        self.group_run_pop_test()
        print("\n\n======================================\nRunning max iterations tests...")
        self.group_run_iteration_test()
        print("\n\n======================================\nRunning explore value tests...")
        self.group_run_explore_test()
        print("\n\n======================================\nRunning mutation probability value tests...")
        self.group_run_mutation_prob_test()
        print("\n\n======================================\nRunning mutation range value tests...")
        self.group_run_mutation_range_test()

    def group_run_pop_test(self):

        for i in range(len(self.group_population_sizes)):
            total_run_time = 0
            mean_run_time = 0
            total_iters_run = 0
            mean_iters_run = 0
            total_successes = 0
            for j in range(self.group_sample_size):
                pop_test = Test(test_id=self.test_id, test_population_size=self.group_population_sizes[i])
                self.test_id += 1
                pop_test.test_run_genetic()
                total_run_time += pop_test.test_run_time
                total_iters_run += pop_test.test_iters_run
                if pop_test.test_optimal_found:
                    total_successes += 1

            mean_run_time = round((total_run_time * 1000) / self.group_sample_size, 2)
            mean_iters_run = round(total_iters_run / self.group_sample_size, 1)
            mean_success_rate = round((total_successes / self.group_sample_size) * 100, 2)
            self.group_results.append({"test_name": "population",
                                       "test_population_size": pop_test.test_population_size,
                                       "test_max_iters": pop_test.test_max_iters,
                                       "test_mutation_prob": pop_test.test_mutation_prob,
                                       "test_mutation_range": pop_test.test_mutation_range,
                                       "test_explore_rate": pop_test.test_explore_rate,
                                       "test_max_children": pop_test.test_max_children,
                                       "total_run_time": int(total_run_time * 1000),
                                       "mean_run_time": mean_run_time,
                                       "mean_iterations_run": mean_iters_run,
                                       "mean_success_rate": mean_success_rate,
                                       })
            print("--------------------------")
            print("Population: " + str(pop_test.test_population_size))
            print("Run Time: " + str(int(total_run_time * 1000)) + " ms")
            print("Avg. Iters Run: " + str(round(mean_iters_run, 1)))
            print("Avg. Success Rate: " + str(mean_success_rate) + "%")


    def group_run_iteration_test(self):
        for i in range(len(self.group_max_iterations)):
            total_run_time = 0
            mean_run_time = 0
            total_iters_run = 0
            mean_iters_run = 0
            total_successes = 0
            for j in range(self.group_sample_size):
                pop_test = Test(test_id=self.test_id, test_max_iters=self.group_max_iterations[i])
                self.test_id += 1
                pop_test.test_run_genetic()
                total_run_time += pop_test.test_run_time
                total_iters_run += pop_test.test_iters_run
                if pop_test.test_optimal_found:
                    total_successes += 1

            mean_run_time = round((total_run_time * 1000) / self.group_sample_size, 2)
            mean_iters_run = round(total_iters_run / self.group_sample_size, 1)
            mean_success_rate = round((total_successes / self.group_sample_size) * 100, 2)
            self.group_results.append({"test_name": "iteration",
                                       "test_population_size": pop_test.test_population_size,
                                       "test_max_iters": pop_test.test_max_iters,
                                       "test_mutation_prob": pop_test.test_mutation_prob,
                                       "test_mutation_range": pop_test.test_mutation_range,
                                       "test_explore_rate": pop_test.test_explore_rate,
                                       "test_max_children": pop_test.test_max_children,
                                       "total_run_time": int(total_run_time * 1000),
                                       "mean_run_time": mean_run_time,
                                       "mean_iterations_run": mean_iters_run,
                                       "mean_success_rate": mean_success_rate,
                                       })
            print("--------------------------")
            print("Max Iterations: " + str(pop_test.test_max_iters))
            print("Run Time: " + str(int(total_run_time * 1000)) + " ms")
            print("Avg. Iters Run: " + str(round(mean_iters_run, 1)))
            print("Avg. Success Rate: " + str(mean_success_rate) + "%")

    def group_run_explore_test(self):
        for i in range(len(self.group_explore_rates)):
            total_run_time = 0
            mean_run_time = 0
            total_iters_run = 0
            mean_iters_run = 0
            total_successes = 0
            for j in range(self.group_sample_size):
                pop_test = Test(test_id=self.test_id, test_explore_rate=self.group_explore_rates[i])
                self.test_id += 1
                pop_test.test_run_genetic()
                total_run_time += pop_test.test_run_time
                total_iters_run += pop_test.test_iters_run
                if pop_test.test_optimal_found:
                    total_successes += 1

            mean_run_time = round((total_run_time * 1000) / self.group_sample_size, 2)
            mean_iters_run = round(total_iters_run / self.group_sample_size, 1)
            mean_success_rate = round((total_successes / self.group_sample_size) * 100, 2)
            self.group_results.append({"test_name": "explore",
                                       "test_population_size": pop_test.test_population_size,
                                       "test_max_iters": pop_test.test_max_iters,
                                       "test_mutation_prob": pop_test.test_mutation_prob,
                                       "test_mutation_range": pop_test.test_mutation_range,
                                       "test_explore_rate": pop_test.test_explore_rate,
                                       "test_max_children": pop_test.test_max_children,
                                       "total_run_time": int(total_run_time * 1000),
                                       "mean_run_time": mean_run_time,
                                       "mean_iterations_run": mean_iters_run,
                                       "mean_success_rate": mean_success_rate,
                                       })
            print("--------------------------")
            print("Explore Rate: " + str(pop_test.test_explore_rate))
            print("Run Time: " + str(int(total_run_time * 1000)) + " ms")
            print("Avg. Iters Run: " + str(round(mean_iters_run, 1)))
            print("Avg. Success Rate: " + str(mean_success_rate) + "%")

    def group_run_mutation_prob_test(self):
        for i in range(len(self.group_mutation_probs)):
            total_run_time = 0
            mean_run_time = 0
            total_iters_run = 0
            mean_iters_run = 0
            total_successes = 0
            for j in range(self.group_sample_size):
                pop_test = Test(test_id=self.test_id, test_mutation_prob=self.group_mutation_probs[i])
                self.test_id += 1
                pop_test.test_run_genetic()
                total_run_time += pop_test.test_run_time
                total_iters_run += pop_test.test_iters_run
                if pop_test.test_optimal_found:
                    total_successes += 1

            mean_run_time = round((total_run_time * 1000) / self.group_sample_size, 2)
            mean_iters_run = round(total_iters_run / self.group_sample_size, 1)
            mean_success_rate = round((total_successes / self.group_sample_size) * 100, 2)
            self.group_results.append({"test_name": "mutation_prob",
                                       "test_population_size": pop_test.test_population_size,
                                       "test_max_iters": pop_test.test_max_iters,
                                       "test_mutation_prob": pop_test.test_mutation_prob,
                                       "test_mutation_range": pop_test.test_mutation_range,
                                       "test_explore_rate": pop_test.test_explore_rate,
                                       "test_max_children": pop_test.test_max_children,
                                       "total_run_time": int(total_run_time * 1000),
                                       "mean_run_time": mean_run_time,
                                       "mean_iterations_run": mean_iters_run,
                                       "mean_success_rate": mean_success_rate,
                                       })
            print("--------------------------")
            print("Mutation Probability: " + str(pop_test.test_mutation_prob))
            print("Run Time: " + str(int(total_run_time * 1000)) + " ms")
            print("Avg. Iters Run: " + str(round(mean_iters_run, 1)))
            print("Avg. Success Rate: " + str(mean_success_rate) + "%")

    def group_run_mutation_range_test(self):
        for i in range(len(self.group_mutation_ranges)):
            total_run_time = 0
            mean_run_time = 0
            total_iters_run = 0
            mean_iters_run = 0
            total_successes = 0
            for j in range(self.group_sample_size):
                pop_test = Test(test_id=self.test_id, test_mutation_range=self.group_mutation_ranges[i])
                self.test_id += 1
                pop_test.test_run_genetic()
                total_run_time += pop_test.test_run_time
                total_iters_run += pop_test.test_iters_run
                if pop_test.test_optimal_found:
                    total_successes += 1

            mean_run_time = round((total_run_time * 1000) / self.group_sample_size, 2)
            mean_iters_run = round(total_iters_run / self.group_sample_size, 1)
            mean_success_rate = round((total_successes / self.group_sample_size) * 100, 2)
            self.group_results.append({"test_name": "mutation_range",
                                       "test_population_size": pop_test.test_population_size,
                                       "test_max_iters": pop_test.test_max_iters,
                                       "test_mutation_prob": pop_test.test_mutation_prob,
                                       "test_mutation_range": pop_test.test_mutation_range,
                                       "test_explore_rate": pop_test.test_explore_rate,
                                       "test_max_children": pop_test.test_max_children,
                                       "total_run_time": int(total_run_time * 1000),
                                       "mean_run_time": mean_run_time,
                                       "mean_iterations_run": mean_iters_run,
                                       "mean_success_rate": mean_success_rate,
                                       })
            print("--------------------------")
            print("Mutation Range: " + str(pop_test.test_mutation_range))
            print("Run Time: " + str(int(total_run_time * 1000)) + " ms")
            print("Avg. Iters Run: " + str(round(mean_iters_run, 1)))
            print("Avg. Success Rate: " + str(mean_success_rate) + "%")


class Test:

    # Set Defaults for Test
    # This replaces the need for a separate Parameters class
    # Default Test settings can be changed when instantiating a Test object
    def __init__(self,
                 test_id=0,
                 test_population_size=50,
                 test_max_iters=100,
                 test_explore_rate=0.1,
                 test_max_variance=0.01,
                 test_mutation_prob=0.1,
                 test_mutation_range=0.2,
                 test_child_multiple=1,
                 ):
        self.test_id = test_id
        self.test_population_size = test_population_size
        self.test_max_iters = test_max_iters
        self.test_explore_rate = test_explore_rate
        self.test_max_variance = test_max_variance
        self.test_mutation_prob = test_mutation_prob
        self.test_mutation_range = test_mutation_range
        self.test_max_children = test_child_multiple * test_population_size
        self.test_run_time = 0
        self.test_iters_run = 0
        self.test_optimal_found = False
        self.test_best_solution = Individual()
        self.test_best_cost = np.infty
        self.test_results = []
        self.problem = Problem()

    def test_choose_pair_from(self, population):
        max_value = len(population) - 1
        index1 = np.random.randint(max_value)
        index2 = np.random.randint(max_value)
        if (index1 == index2) & (max_value > 1):
            return self.test_choose_pair_from(population)
        else:
            return index1, index2

    def test_run_genetic(self):
        start_time = time.perf_counter()
        cost_function = self.problem.cost_function
        population = []

        # Initialize the population set and get the most "fit" individual
        for i in range(self.test_population_size):
            new_individual = Individual(self.problem)
            population.append(new_individual)

            if new_individual.cost < self.test_best_cost:
                self.test_best_solution = deepcopy(new_individual)
                self.test_best_cost = new_individual.cost

        # Start a series of "breeding" and mutation iterations
        # until either an acceptable solution is found or up to
        # the tests maximum  allowed iterations
        for j in range(self.test_max_iters):
            children = []
            self.test_iters_run += 1
            while len(children) < self.test_max_children:
                parent1_index, parent2_index = self.test_choose_pair_from(population)
                parent1 = population[parent1_index]
                parent2 = population[parent2_index]

                # Use crossover to produce 2 children
                child1, child2 = parent1.crossover(parent2, self.test_explore_rate)

                # Mutate these children
                child1.mutate(self.test_mutation_prob, self.test_mutation_range)
                child2.mutate(self.test_mutation_prob, self.test_mutation_range)

                # calculate costs for these children
                child1.cost = cost_function(child1.chromosome)
                child2.cost = cost_function(child2.chromosome)

                # add to the children population
                children.append(child1)
                children.append(child2)

            # print ("New Children: " + str(len(children)))
            population += children
            population = sorted(population, key=lambda x: x.cost)

            # Cull the population to top results fitting population size
            population = population[0:self.test_population_size]

            # Update best solution
            if population[0].cost < self.test_best_cost:
                self.test_best_solution = deepcopy(population[0])
                self.test_best_cost = self.test_best_solution.cost

                if self.test_best_cost <= self.test_max_variance:
                    self.test_optimal_found = True
                    break

        end_time = time.perf_counter()
        self.test_run_time = end_time - start_time


def show_results(result_set):
    x_axis_population = []
    y_axis_population_time = []
    y_axis_population_success = []
    x_axis_iteration = []
    y_axis_iteration_time = []
    y_axis_iteration_success = []
    x_axis_explore = []
    y_axis_explore_time = []
    y_axis_explore_success = []
    x_axis_mutation_prob = []
    y_axis_mutation_prob_time = []
    y_axis_mutation_prob_success = []
    x_axis_mutation_range = []
    y_axis_mutation_range_time = []
    y_axis_mutation_range_success = []

    for i in result_set:
        if i["test_name"] == "population":
            x_axis_population.append(i["test_population_size"])
            y_axis_population_time.append(i["total_run_time"])
            y_axis_population_success.append(i["mean_success_rate"])
        elif i["test_name"] == "iteration":
            x_axis_iteration.append(i["test_max_iters"])
            y_axis_iteration_time.append(i["total_run_time"])
            y_axis_iteration_success.append(i["mean_success_rate"])
        elif i["test_name"] == "explore":
            x_axis_explore.append(i["test_explore_rate"])
            y_axis_explore_time.append(i["total_run_time"])
            y_axis_explore_success.append(i["mean_success_rate"])
        elif i["test_name"] == "mutation_prob":
            x_axis_mutation_prob.append(i["test_mutation_prob"])
            y_axis_mutation_prob_time.append(i["total_run_time"])
            y_axis_mutation_prob_success.append(i["mean_success_rate"])
        elif i["test_name"] == "mutation_range":
            x_axis_mutation_range.append(i["test_mutation_range"])
            y_axis_mutation_range_time.append(i["total_run_time"])
            y_axis_mutation_range_success.append(i["mean_success_rate"])

    # Plot the data
    max_time = max(y_axis_population_time + y_axis_iteration_time + y_axis_explore_time + y_axis_mutation_prob_time + y_axis_mutation_range_time) + 1000
    max_time = round(max_time, -3)
    print(y_axis_population_time)
    print(y_axis_iteration_time)
    print(y_axis_explore_time)
    print(y_axis_mutation_prob_time)
    print(y_axis_mutation_range_time)

    show_chart(x_axis_population, y_axis_population_time, y_axis_population_success, 'Population', max_time)
    show_chart(x_axis_iteration, y_axis_iteration_time, y_axis_iteration_success, 'Iteration', max_time)
    show_chart(x_axis_explore, y_axis_explore_time, y_axis_explore_success, 'Explore Rate', max_time)
    show_chart(x_axis_mutation_prob, y_axis_mutation_prob_time, y_axis_mutation_prob_success, 'Mutation Probability', max_time)
    show_chart(x_axis_mutation_range, y_axis_mutation_range_time, y_axis_mutation_range_success, 'Mutation Range', max_time)


def show_chart(test_results, data1, data2, xlabel, max_time):
    fig, ax1 = plt.subplots()

    red = 'tab:red'
    green = 'tab:green'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Run Time (ms)', color=red)
    ax1.plot(test_results, data1, color=red)
    ax1.set_yticks(np.arange(0, max_time, max_time/10))
    ax1.tick_params(axis='y', labelcolor=red)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Success Rate %', color=green)
    ax2.plot(test_results, data2, color=green)
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.tick_params(axis='y', labelcolor=green)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def main():
    sample_size = int(input("How many times should each test be run?"))
    my_test_group = TestGroup(group_sample_size=sample_size,
                              group_population_sizes=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                              group_max_iterations=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                              group_explore_rates=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
                              group_mutation_probs=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
                              group_mutation_ranges=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                              )
    my_test_group.group_run_tests()
    show_results(my_test_group.group_results)


main()
