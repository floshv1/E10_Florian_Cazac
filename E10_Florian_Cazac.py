import numpy as np
import csv

class BranchAndBound:
    def __init__(self, profits, weights, capacity):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity
        self.n = len(profits)
        self.max_profit = 0
        self.best_solution = [0] * self.n

    def upper_bound(self, level, profit, weight):
        if weight >= self.capacity:
            return 0

        bound = profit
        total_weight = weight
        for i in range(level, self.n):
            if total_weight + self.weights[i] <= self.capacity:
                total_weight += self.weights[i]
                bound += self.profits[i]
            else:
                bound += (self.profits[i] / self.weights[i]) * (self.capacity - total_weight)
                break

        return bound

    def branch_and_bound(self, level, profit, weight, solution):
        if weight > self.capacity:
            return

        if profit > self.max_profit:
            self.max_profit = profit
            self.best_solution = solution[:]

        if level == self.n:
            return

        # Include the current item
        solution[level] = 1
        self.branch_and_bound(
            level + 1,
            profit + self.profits[level],
            weight + self.weights[level],
            solution
        )

        # Exclude the current item
        solution[level] = 0
        if self.upper_bound(level + 1, profit, weight) > self.max_profit:
            self.branch_and_bound(level + 1, profit, weight, solution)

    def solve(self):
        self.branch_and_bound(0, 0, 0, [0] * self.n)
        return self.max_profit, self.best_solution


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

    # Solve using Branch and Bound
    bb_solver = BranchAndBound(profits, weights, total_capacity)
    max_profit, selected_items = bb_solver.solve()

    print("Maximum Profit:", max_profit)
    print("Selected Items:", selected_items)
    print("Items selected:", [i + 1 for i, x in enumerate(selected_items) if x == 1])
