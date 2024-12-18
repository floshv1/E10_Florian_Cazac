import numpy as np
import csv

import numpy as np
from queue import PriorityQueue

class Node:
    def __init__(self, level, profit, weight, bound, items):
        self.level = level  # Depth in the decision tree
        self.profit = profit  # Total profit at this node
        self.weight = weight  # Total weight at this node
        self.bound = bound  # Upper bound of profit
        self.items = items  # Items taken

    def __lt__(self, other):
        return self.bound > other.bound  # Priority by bound (max profit first)

def calculate_bound(node, n, capacity, profits, weights):
    if node.weight >= capacity:
        return 0  # No feasible solution

    bound = node.profit
    total_weight = node.weight

    for i in range(node.level + 1, n):
        if total_weight + weights[i] <= capacity:
            total_weight += weights[i]
            bound += profits[i]
        else:
            bound += (capacity - total_weight) * (profits[i] / weights[i])
            break

    return bound

def branch_and_bound(capacity, profits, weights):
    n = len(profits)
    pq = PriorityQueue()
    
    root = Node(level=-1, profit=0, weight=0, bound=0, items=[0] * n)
    root.bound = calculate_bound(root, n, capacity, profits, weights)
    pq.put(root)

    max_profit = 0
    best_items = []

    while not pq.empty():
        node = pq.get()

        if node.bound > max_profit:
            level = node.level + 1

            # Include the current item
            if level < n:
                left_child = Node(level=level, 
                                  profit=node.profit + profits[level],
                                  weight=node.weight + weights[level],
                                  bound=0,
                                  items=node.items[:])
                left_child.items[level] = 1

                if left_child.weight <= capacity and left_child.profit > max_profit:
                    max_profit = left_child.profit
                    best_items = left_child.items

                left_child.bound = calculate_bound(left_child, n, capacity, profits, weights)
                if left_child.bound > max_profit:
                    pq.put(left_child)

            # Exclude the current item
            right_child = Node(level=level, 
                               profit=node.profit,
                               weight=node.weight,
                               bound=0,
                               items=node.items[:])

            right_child.bound = calculate_bound(right_child, n, capacity, profits, weights)
            if right_child.bound > max_profit:
                pq.put(right_child)

    return max_profit, best_items

def solve_integer_problem(input_table, total_capacity):
    # Extract profits and weights from the input table
    profits = input_table[:, 0]
    weights = input_table[:, 1]

    max_profit, selected_items = branch_and_bound(total_capacity, profits, weights)

    return max_profit, selected_items


def load_data_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        data = np.array([[float(cell) for cell in row] for row in reader])
    return data


if __name__ == "__main__":
    # Load data from a CSV file
    file_path = "table.csv"  # Replace with the path to your CSV file
    input_table = load_data_from_csv(file_path)

    profits = input_table[:, 0]
    weights = input_table[:, 1]
    total_capacity = 100  # Total researcher days available

    max_profit, selected_items = solve_integer_problem(input_table, total_capacity)

    print("Maximum Profit:", max_profit)
    print("Selected Items:", selected_items)
