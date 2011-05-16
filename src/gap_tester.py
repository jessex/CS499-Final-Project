from sys import argv, exit, maxint
import random, time, math
from copy import deepcopy

ELITISM = 0.00 #percentage of population to definitely carry over to next gen
TOURNAMENT_K = 8 #amount of candidates to be chosen for tournament selection
MAX_GENERATIONS = 250 #maximum amount of generations to be run through
POP_SIZE = 50

subset_sizes = [4,3,2,4,2,6,3,4,6,8]

class Partition:

	def __init__(self, partition, fitness):
		self.partition = partition #dictionary of subsets to local fitness vals
		self.fitness = fitness #overall partition fitness
	
	def __repr__(self):
		str = "Partition with fitness %.8f:\n" % self.fitness
		for subset in self.partition.keys():
			str += "Subset:\n"
			for i in range(len(subset)-1):
				str += SCHEDULES[subset[i]][1] + ", "
			str += SCHEDULES[subset[i+1]][1] + "\n"
		return str
	

# ****************************** SETUP FUNCTIONS ******************************

def read_file(filename, schedule_length):
	"""Reads in the text from the file with the passed file name and parses out 
	a list of schedules of length schedule_length."""
	global STRING_LENGTH
	f = open(filename)
	STRING_LENGTH = schedule_length
	schedules = []
	for line in f:
		if not line.lstrip():
			continue
		else:
			portions = line.rsplit(",") #schedule,name
			if len(portions[0]) > schedule_length:
				portions[0] = portions[0][0:schedule_length]
			schedules.append((portions[0], portions[1].rstrip()))
	return schedules
	
def initialize_population(pop_size, schedules, k):
	"""Generates pop_size total random partitions of schedules into subsets of  
	size k and returns the population of partitions."""
	global SUBSET_COUNT, SUBSET_SIZE, HALFWAY
	SUBSET_SIZE = k
	HALFWAY = k/2
	SUBSET_COUNT = len(schedules) / SUBSET_SIZE
	population = []
	while len(population) < pop_size:
		population.append(random_partition(schedules))
	return population
	
def random_partition(schedules):
	"""Takes in a list of schedules and k, the subset size, and returns a  
	random partioning of the schedules into subset groups."""
	indices = [i for i in range(len(schedules))]
	subset = []
	partition = {}
	while indices:
		subset.append(indices.pop(random.randint(0, len(indices)-1)))
		if len(subset) == SUBSET_SIZE:
			partition[tuple(subset)] = 0
			subset = []
	if subset:
		partition[tuple(subset)] = 0
	return Partition(partition, 0)
	
# ***************************** GENETIC ALGORITHM *****************************
	
def get_subset_fitness(subset, schedules):
	"""Calculates and returns the fitness for a subset of schedules. Takes in 
	the subset to measure for fitness and the list of all schedules."""
	strings = [schedules[i][0] for i in subset]
	total = 0
	for i in range(STRING_LENGTH):
		ones = 0
		for s in strings:
			if s[i] == "1":
				ones += 1
		if ones <= HALFWAY:
			total += ones
		else:
			total += SUBSET_SIZE - ones
	return total
	
def get_partition_fitness(partition):
	"""Calculates the local fitness for each individual subset in the partition 
	and then returns the Euclidean norm (square root of the sum of the squared 
	subset fitness values) as the total fitness of the partition."""
	locals = [get_subset_fitness(subset, SCHEDULES)**2 for subset in partition.partition]
	return math.sqrt(sum(locals))
	
def measure_population(population):
	"""Updates the fitness score for each partition in the population."""
	for partition in population:
		partition.fitness = get_partition_fitness(partition)

def sort_by_fitness(population):
	"""Sorts the partition individuals in the population by fitness and returns 
	the sorted population list."""
	return sorted(population, key=lambda partition: partition.fitness)
	
def segregate_elite(sorted_population, percentage):
	"""Pulls the elite percentage of the population from sorted_population and 
	returns it. sorted_population is modified in place, so the elite subset is  
	ripped from the rest of the population."""
	elite = []
	total = int(len(sorted_population) * percentage)
	for i in range(total):
		elite.append(sorted_population.pop(0))
	return elite
	
def breed_parents(x, y):
	"""Takes in two parent partitions and crossbreeds them to produce a child 
	partition. Adds subsets from the parents in order of fitness until it has 
	the correct number of subsets. Then checks for duplicate entries (some 
	entries may be present no more than twice) and removes them from the less 
	fit of the two subsets that the given duplicate is present within. Then 
	fills the subsets which just had a duplicate removed with those entries 
	which were never added to the child to begin with. The new child is then 
	returned."""
	partition = []
	fits = []
	#choose most fit subsets from parents until we have the right amount
	x_subs = sorted(x.partition.items(), key=lambda pair: pair[1])
	y_subs = sorted(y.partition.items(), key=lambda pair: pair[1])
	i = j = 0
	added = [0 for a in range(len(SCHEDULES))] #schedules added into child
	while len(partition) < SUBSET_COUNT:
		if x_subs[i][1] < y_subs[j][1]: #next best x greater than next best y
			partition.append(x_subs[i][0])
			fits.append(x_subs[i][1])
			for index in x_subs[i][0]:
				added[index] += 1
			i += 1
		elif x_subs[i][1] > y_subs[j][1]: #next best y greater than next best x
			partition.append(y_subs[j][0])
			fits.append(y_subs[j][1])
			for index in y_subs[i][0]:
				added[index] += 1
			j += 1
		else: #next best x equal to next best y, so coin flip
			if random.randint(0, 1):
				partition.append(x_subs[i][0])
				fits.append(x_subs[i][1])
				for index in x_subs[i][0]:
					added[index] += 1
				i += 1
			else:
				partition.append(y_subs[j][0])
				fits.append(y_subs[j][1])
				for index in y_subs[j][0]:
					added[index] += 1
				j += 1
	#check for duplicate and non-represented schedules in child
	zeroes = []
	for i in range(len(added)): #mark those which still have to be added in
		if added[i] == 0:
			zeroes.append(i)
	for i in range(len(added)):
		if added[i] == 2: #check for those which are duplicated
			subsets = []
			checks = []
			for j in range(len(partition)): #get the subsets with the duplicate
				if i in partition[j]:
					subsets.append(partition[j])
					checks.append(fits[j])
					if len(subsets) == 2:
						break
			#pick the subset with the worse fitness to remove duplicate from
			if checks[0] >= checks[1]:
				partition.remove(subsets[0])
				a = list(subsets[0])
			else:
				partition.remove(subsets[1])
				a = list(subsets[1])
			#place random choice of one of the not yet added into removed subset
			a.remove(i) #remove duplicate first
			k = zeroes.pop(random.randint(0, len(zeroes)-1))
			a.append(k)
			partition.append(tuple(a)) #add new valid subset to partition
			added[k] = 1 #up from 0 back to 1
			added[i] = 1 #down from 2 back to 1
	
	subsets = {}
	for subset in partition: #convert list of subsets back into dict and return
		subsets[subset] = 0
	return Partition(subsets, 0)
	
def tournament_select(k, population):
	"""Run a deterministic tournament selection which chooses the candidate with 
	the best fitness from a random subset of k individuals from population."""
	upper = len(population) - 1
	candidates = []
	if k > len(population): #very small population size or huge tournament size
		k = len(population)
	for i in range(k):
		candidates.append(population[random.randint(0,upper)])
	return sort_by_fitness(candidates)[0] #choose best with probability=1
	
def recombinate_population(parents):
	"""Performs crossover breeding of the parents population until we have mated 
	a total amount of children equal to the original amount of parents. The 
	selection procedure for parents is a deterministic tournament selection."""
	goal = len(parents)
	children = []
	while len(children) < goal:
		x = tournament_select(TOURNAMENT_K, parents)
		y = tournament_select(TOURNAMENT_K, parents)
		while y == x: #make sure we do not breed x with itself
			y = tournament_select(TOURNAMENT_K, parents)
		children.append(breed_parents(x, y))
	return children
	
def mutate_partition(partition):
	"""Performs a single mutation on the partition, which consists of swapping 
	two randomly chosen schedules from two randomly chosen subsets."""
	#choose random subsets to perform swap mutation
	keys = partition.partition.keys()
	subset_a = keys[random.randint(0, len(keys)-1)]
	subset_b = keys[random.randint(0, len(keys)-1)]
	while subset_a == subset_b: #make sure we actually mutate
		subset_b = keys[random.randint(0, len(keys)-1)]
	#remove older subsets from partition dictionary
	del partition.partition[subset_a]
	del partition.partition[subset_b]
	
	#swap random schedule from a with random schedule from b
	subset_a = list(subset_a)
	subset_b = list(subset_b)
	sched_a = subset_a.pop(random.randint(0, len(subset_a)-1))
	sched_b = subset_b.pop(random.randint(0, len(subset_b)-1))
	subset_a.append(sched_b)
	subset_b.append(sched_a)
	
	#add new subsets back to partition dictionary
	partition.partition[tuple(subset_a)] = 0
	partition.partition[tuple(subset_b)] = 0	
	
def mutate_population(population):
	"""Conducts mutation on the entire population, with greater mutation being 
	performed on those with poorer fitness. The most fit individuals have fewer 
	mutation steps and less chance of mutation at each step. The less fit 
	individuals have more mutation steps and higher chance of mutation at each 
	step. The mutation is performed in place on population."""
	size = len(population)
	first_upper = 0.10 * size
	second_upper = 0.40 * size
	for i in range(size):
		if i >= second_upper: #final 60 percent
			for j in range(10): #10 mutation steps
				if random.random() < 0.5: #each step has probability=0.5
					mutate_partition(population[i])
		elif first_upper <= i < second_upper: #previous 30 percent
			for j in range(4): #4 mutation steps
				if random.random() < 0.5: #each step has probability=0.5
					mutate_partition(population[i])
		else: #first 10 percent
			for j in range(1):
				if random.random() < 0.1: #one mutation step with probability=0.1
					mutate_partition(population[i])
	
def iterate_generation(population):
	"""Runs one generation of the population of candidates and returns the new, 
	updated population. Runs through elitism, crossover, fitness evaluation, 
	mutation and another fitness evaluation."""
	#pull out the elite and crossbreed rest of population to rebuild
	elite = segregate_elite(population, ELITISM)
	children = recombinate_population(population)
	
	measure_population(children) #recalc fitness for children
	population = elite + children #recombine elite with new children
	population = sort_by_fitness(population) #re-sort total population
	
	mutate_population(population) #mutate according to position in population
	measure_population(population) #recalc fitness after mutation
	population = sort_by_fitness(population) #re-sort total population again
	return population

def run_genetic(subset_size):
	"""Runs the genetic algorithm through the test runs."""
	#initialize a random population and set up for run
	population = initialize_population(POP_SIZE, SCHEDULES, subset_size)
	BEST_SO_FAR = Partition({}, maxint)
	BEST_SO_FAR_GEN = 0
	generation = 0
	beginning = time.time() #start the clock
	
	#iterate through generations until termination condition is met
	while generation < MAX_GENERATIONS:
		population = iterate_generation(population)
		generation += 1
		#print "Generation %d: best fitness = %.8f" % (generation, population[0].fitness)
		if population[0].fitness < BEST_SO_FAR.fitness:
			BEST_SO_FAR = deepcopy(population[0])
			BEST_SO_FAR_GEN = generation
		if population[0].fitness == 0: #found a literally perfect partition
			break
			
	end = time.time() #stop the clock, report results
	print "\nBest Partition:\n%s" % BEST_SO_FAR
	print "First found at generation %d (%.4f sec runtime)" % (BEST_SO_FAR_GEN, (end - beginning))
	
# **************************** EVOLUTION STRATEGY *****************************
	
def run_ES(subset_size):
	"""Runs the evolution strategy through the test runs."""
	global SUBSET_SIZE, HALFWAY, SUBSET_COUNT
	SUBSET_SIZE = subset_size
	HALFWAY = subset_size/2
	SUBSET_COUNT = len(SCHEDULES) / SUBSET_SIZE
	
	partition = random_partition(SCHEDULES)
	partition.fitness = get_partition_fitness(partition)
	
	generation = 0
	BEST_SO_FAR = Partition({}, maxint)
	BEST_SO_FAR_GEN = 0
	beginning = time.time()
	
	while generation < MAX_GENERATIONS:
		generation += 1
		#old_partition = deepcopy(partition)
		children = []
		
		for i in range(5): #5 children to mutate from parent
			parent = deepcopy(partition)
			mutate_partition(partition)
			partition.fitness = get_partition_fitness(partition)
			children.append(partition)
			partition = deepcopy(parent)
			
		best_fitness = maxint
		for child in children: #find best child
			if partition.fitness < best_fitness:
				best_fitness = partition.fitness
				partition = deepcopy(child)
			
		if partition.fitness < BEST_SO_FAR.fitness: #partition best so far
			BEST_SO_FAR = deepcopy(partition)
			BEST_SO_FAR_GEN = generation
			
	end = time.time()
	print "\nBest Partition:\n%s" % BEST_SO_FAR
	print "First found at generation %d (%.4f sec runtime)" % (BEST_SO_FAR_GEN, (end - beginning))

# ******************************* RANDOM SEARCH *******************************

def run_random(subset_size):
	"""Runs the random search through the test runs."""
	global SUBSET_SIZE, HALFWAY, SUBSET_COUNT
	SUBSET_SIZE = subset_size
	HALFWAY = subset_size/2
	SUBSET_COUNT = len(SCHEDULES) / SUBSET_SIZE
	
	partition = random_partition(SCHEDULES)
	partition.fitness = get_partition_fitness(partition)
	
	generation = 0
	BEST_SO_FAR = Partition({}, maxint)
	BEST_SO_FAR_GEN = 0
	beginning = time.time()
	
	while generation < MAX_GENERATIONS:
		generation += 1
		new_partition = random_partition(SCHEDULES)
		new_partition.fitness = get_partition_fitness(new_partition)
		
		if new_partition.fitness <= partition.fitness: #random was better
			partition = deepcopy(new_partition)
			
		if partition.fitness < BEST_SO_FAR.fitness: #partition best so far
			BEST_SO_FAR = deepcopy(partition)
			BEST_SO_FAR_GEN = generation
	
	end = time.time() #stop the clock, report results
	print "\nBest Partition:\n%s" % BEST_SO_FAR
	print "First found at generation %d (%.4f sec runtime)" % (BEST_SO_FAR_GEN, (end - beginning))

# ***************************** GREEDY ALGORITHM ******************************

def run_greedy(subset_size):
	"""Runs the greedy algorithm through the test runs."""
	global SUBSET_SIZE, HALFWAY, SUBSET_COUNT
	SUBSET_SIZE = subset_size
	HALFWAY = subset_size/2
	SUBSET_COUNT = len(SCHEDULES) / SUBSET_SIZE
	beginning = time.time()
	
	subsets = []
	current_subset = []
	current_subset.append(get_random_schedule(subsets))
	
	for i in range(len(SCHEDULES)-1):
		if len(current_subset) == SUBSET_SIZE:
			subsets.append(tuple(current_subset))
			current_subset = []
			current_subset.append(get_random_schedule(subsets))
		else:
			current_subset.append(get_best_addition(current_subset, subsets))
	subsets.append(tuple(current_subset))
	
	partitions = {}
	for subset in subsets:
		partitions[subset] = 0
	partition = Partition(partitions, maxint)
	partition.fitness = get_partition_fitness(partition)
	
	end = time.time()
	print "\nBest Partition:\n%s" % partition
	print "(%.4f sec runtime)" % (end - beginning)
	
def get_best_addition(subset, subsets):
	"""Returns the best possible schedule which can be added to the given subset
	that is not already in any of the subsets."""
	temporary = deepcopy(subset)
	best_score = maxint
	best_addition = None
	
	for i in range(len(SCHEDULES)):
		valid = True
		for j in subsets:
			if i in j:
				valid = False
		if i not in subset and valid:
			temporary.append(i)
			temp_score = get_subset_fitness(temporary, SCHEDULES)
			if temp_score < best_score:
				best_score = temp_score
				best_addition = i
				
			temporary = deepcopy(subset)
			
	return best_addition
	
def get_random_schedule(subsets):
	"""Returns a random schedule which is not contained in any of the given
	subsets."""
	not_yet = []
	for i in range(len(SCHEDULES)):
		not_in = True
		for j in subsets:
			if i in j:
				not_in = False
		if not_in:
			not_yet.append(i)
	return not_yet[random.randint(0, len(not_yet)-1)]
		
# ******************************* MAIN/TESTING ********************************

if __name__ == "__main__":
	if len(argv) < 2:
		exit("Error: Please provide an algorithm option")
	choice = argv[1]
		
	for i in range(10): #10 different tests
		print "RUNNING TEST", i+1
		for j in range(10): #10 runs a piece
			text = "TEST " + str(i+1) + ": Run " + str(j+1)
			print text
			
			subset_size = subset_sizes[i]
			filename = "test" + str(i+1) + ".txt"
			SCHEDULES = read_file(filename, 48)
			
			if choice == "-ge":
				run_genetic(subset_size)
			elif choice == "-gr":
				run_greedy(subset_size)
			elif choice == "-r":
				run_random(subset_size)
			elif choice == "-e":
				run_ES(subset_size)
			else:
				exit("Error: Please choose -ge for Genetic, -gr for Greedy, -r for Random or -e for Evolution Strategy")
		
		print "\n\n"
	