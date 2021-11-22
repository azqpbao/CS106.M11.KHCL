from ortools.linear_solver import _pywraplp
from ortools.algorithms import pywrapknapsack_solver
import seaborn as sns
import matplotlib.pyplot as plt
import time
import data
import os
def main():
	#from 1 to 12
	# print('Input folder to run (1->13): ', end = '')
	# folder_to_run = int(input())
	inp = data.input_data() 
	
	check_point = 57

	# for name in range((folder_to_run - 1)*8, (folder_to_run - 1)*8 + 8):
	for name in range(check_point, len(inp)):
		#Declare time
		start = time.time()
		elapsed = 0
		print('File name: ' + inp[name])
		print('Package number: ' + str(name))
		with open(format(os.getcwd())+"/Documents/Knapsack/"+inp[name] + ".kp") as level_file:
			rows = level_file.read().split('\n')
		
		number_items = (int)(rows[1])
		capacities = [(int)(rows[2])]
		values = []
		weights = [[]]
		for i in range(4, number_items + 4):
			x = rows[i].split(" ")[0]
			y = rows[i].split(" ")[1]
			values.append((int)(x))
			weights[0].append((int)(y))
			

		solver = pywrapknapsack_solver.KnapsackSolver(
			pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')


		#Set time limit - seconds
		#3 minutes -> 180 seconds
		solver.set_time_limit(180)

		solver.Init(values, weights, capacities)
		computed_value = solver.Solve()

		packed_items = []
		packed_weights = []
		total_weight = 0
		for i in range(len(values)):
			if solver.BestSolutionContains(i):
				packed_items.append(i)	
				packed_weights.append(weights[0][i])
				total_weight += weights[0][i]

		elapsed = time.time() - start

		with open(format(os.getcwd())+"/Documents/Knapsack/"+"output/Google-OR-Tools/" + "test " + str(name) + ".txt", 'w+') as solver_file: 
			solver_file.write('File name: {}.kp\nExecution time: {} sec\nCapacity = {} \nTotal weight = {} \nTotal value = {} \nNumber of items: {} \n' \
							.format(inp[name], elapsed, capacities[0], total_weight, computed_value, len(packed_items)))
			solver_file.write('Packed items: {} \n'.format(packed_items))
			solver_file.write('Packed weights: {}'.format(len(packed_items)))

		print('Execution time: ' + str(elapsed))
		print()

if __name__ == '__main__':
	main()