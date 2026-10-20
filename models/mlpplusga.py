# Import necessary libraries
import pygad
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

seed = 100

lower = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.0, 1, 0.0]
upper = [10, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 3, 2, 0.3, 1000, 1.0]

def convert_range(values):
    return [l + value * (u - l) for value, l, u in zip(values, lower, upper)]

def decode_chromosome(x):
    x = convert_range(x)
    num_layers = int(round(x[0]))
    hidden_layer_sizes = [int(round(x)) for x in x[1:num_layers]]
    match int(round(x[num_layers+1])):
        case 0:
            activation = 'identity'
        case 1:
            activation = 'logistic'
        case 2:
            activation = 'tanh'
        case _:
            activation = 'relu'
    match int(round(x[num_layers+2])):
        case 0:
            solver = 'lbfgs'
        case 1:
            solver = 'sgd'
        case _:
            solver = 'adam'
    learning_rate = x[num_layers+3].item()
    max_iter = int(round(x[num_layers+4]))
    momentum = x[num_layers+5].item()

    return hidden_layer_sizes, activation, solver, learning_rate, max_iter, momentum

def fitness_func(ga_instance, solution, solution_idx):
    
    hidden_layer_sizes, activation, solver, learning_rate, max_iter, momentum = decode_chromosome(solution)

    # 1. Load and prepare the data
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=seed)

    # 2. Feature Scaling (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Initialize the MLP Classifier
    # Define the model with specific parameters
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        momentum=momentum,
        random_state=seed
    )

    # 4. Train the model
    mlp.fit(X_train_scaled, y_train)

    # 5. Make predictions
    y_pred = mlp.predict(X_test_scaled)

    # 6. Evaluate the model
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    ga = pygad.GA(
        num_generations=50,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=16, 
        init_range_low=0.0,
        init_range_high=1.0,
        gene_space=
        crossover_probability=0.9,
        mutation_probability=0.1,
        random_seed=seed
    )

    ga.run()

    solution, solution_fitness, _ = ga.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))