#########################################################
#                                                       #
#       HYBRID GENETIC ALGORITHM (24.05.2016)           #
#                                                       #
#               R. Lahoz-Beltra                         #
#                                                       #    
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY #
# AND self.fitness FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  #
# THE SOFWTARE CAN BE USED BY ANYONE SOLELY FOR THE     #
# PURPOSES OF EDUCATION AND RESEARCH.                   #
#                                                       #
#########################################################
import math
import random
import numpy as np
import matplotlib.pyplot as plt


class HGA:
    #########################################################
    # ALGORITHM PARAMETERS                                  #
    #########################################################
    N = 50                          # Define here the population size
    Genome = 4                      # Define here the chromosome length
    generation_max = 650            # Define here the maximum number of generations/iterations
    fitness_maximization = True     # true if you want to maximize fitness, false if you want to minimize fitness

    #########################################################
    # PARAMETERS INITIALIZATION                             #
    #########################################################

    def __init__(self, n=50, genome=4, generation_max=650, fitness_maximization=True):
        self.N = n
        self.Genome = genome
        self.generation_max = generation_max
        self.fitness_maximization = self.fitness_maximization
        self.init_variables()

    #########################################################
    # VARIABLES ALGORITHM                                   #
    #########################################################
    def init_variables(self):
        self.popSize = self.N + 1
        self.genomeLength = self.Genome + 1
        self.top_bottom = 3
        self.QuBitZero = np.array([[1], [0]])
        self.QuBitOne = np.array([[0], [1]])
        self.AlphaBeta = np.empty([self.top_bottom])
        self.fitness = np.empty([self.popSize])
        self.probability = np.empty([self.popSize])
        # qpv: quantum chromosome (or population vector, QPV)
        self.qpv = np.empty([self.popSize, self.genomeLength, self.top_bottom])
        self.nqpv = np.empty([self.popSize, self.genomeLength, self.top_bottom])
        # chromosome: classical chromosome
        self.chromosome = np.empty([self.popSize, self.genomeLength], dtype=int)
        self.child1 = np.empty([self.popSize, self.genomeLength, self.top_bottom])
        self.child2 = np.empty([self.popSize, self.genomeLength, self.top_bottom])
        self.best_chrom = np.empty([self.generation_max], dtype=int)

        # Initialization global variables
        self.theta = 0
        self.the_best_chrom = 0
        self.generation = 0

    #########################################################
    # QUANTUM POPULATION INITIALIZATION                     #
    #########################################################
    def Init_population(self):
        # Hadamard gate
        r2 = math.sqrt(2.0)
        h = np.array([[1 / r2, 1 / r2], [1 / r2, -1 / r2]])
        # Rotation Q-gate
        self.theta = 0
        rot = np.empty([2, 2])
        # Initial population array (individual x chromosome)
        i = 1
        j = 1
        for i in range(1, self.popSize):
            for j in range(1, self.genomeLength):
                self.theta = np.random.uniform(0, 1) * 90
                self.theta = math.radians(self.theta)
                rot[0, 0] = math.cos(self.theta)
                rot[0, 1] = -math.sin(self.theta)
                rot[1, 0] = math.sin(self.theta)
                rot[1, 1] = math.cos(self.theta)
                self.AlphaBeta[0] = rot[0, 0] * (h[0][0] * self.QuBitZero[0]) + rot[0, 1] * (h[0][1] * self.QuBitZero[1])
                self.AlphaBeta[1] = rot[1, 0] * (h[1][0] * self.QuBitZero[0]) + rot[1, 1] * (h[1][1] * self.QuBitZero[1])
                # alpha squared
                self.qpv[i, j, 0] = np.around(2 * pow(self.AlphaBeta[0], 2), 2)
                # beta squared
                self.qpv[i, j, 1] = np.around(2 * pow(self.AlphaBeta[1], 2), 2)

    #########################################################
    # SHOW QUANTUM POPULATION                               #
    #########################################################
    def Show_population(self):
        i = 1
        j = 1
        for i in range(1, self.popSize):
            print()
            print()
            print("qpv = ", i, " : ")
            print()
            for j in range(1, self.genomeLength):
                print(self.qpv[i, j, 0], end="")
                print(" ", end="")
            print()
            for j in range(1, self.genomeLength):
                print(self.qpv[i, j, 1], end="")
                print(" ", end="")
        print()

    ##########################################################
    # MAKE A MEASURE                                         #
    ##########################################################
    # p_alpha: self.probability of finding qubit in alpha state    
    def Measure(self, p_alpha):
        for i in range(1, self.popSize):
            print()
            for j in range(1, self.genomeLength):
                if p_alpha <= self.qpv[i, j, 0]:
                    self.chromosome[i, j] = 0
                else:
                    self.chromosome[i, j] = 1
                print(self.chromosome[i, j], " ", end="")
            print()
        print()

    #########################################################
    # fitness COMPARISON                                    #
    #########################################################
    # return true if fitness1 is better then fitness2
    def is_better(self, fitness1, fitness2):
        return (self.fitness_maximization and fitness1 > fitness2) or (not self.fitness_maximization and fitness1 < fitness2)

    # return true if self.fitness1 is better then or equal to self.fitness2
    def is_better_equal(self, fitness1, fitness2):
        return (self.fitness_maximization and fitness1 >= fitness2) or (not self.fitness_maximization and fitness1 <= fitness2)

    def is_worse(self, fitness1, fitness2):
        return not (self.is_better_equal(fitness1, fitness2))

    def is_worse_equal(self, fitness1, fitness2):
        return not (self.is_better(fitness1, fitness2))

    #########################################################
    # FITNESS EVALUATION                                    # 
    #########################################################
    def fitness_evaluation(self, generation):
        i = 1
        j = 1
        fitness_total = 0
        sum_sqr = 0
        fitness_average = 0
        variance = 0
        for i in range(1, self.popSize):
            self.fitness[i] = 0

        #########################################################
        # Define your problem in this section. For instance:    #
        #                                                       #
        # Let f(x)=abs(x-5/2+sin(x)) be a function that takes   #
        # values in the range 0<=x<=15. Within this range f(x)  #
        # has a maximum value at x=11 (binary is equal to 1011) #
        #########################################################
        for i in range(1, self.popSize):
            x = 0
            for j in range(1, self.genomeLength):
                # translate from binary to decimal value
                x = x + self.chromosome[i, j] * pow(2, self.genomeLength - j - 1)
                # replaces the value of x in the function f(x)
                y = np.fabs((x - 5) / (2 + np.sin(x)))
                # the fitness value is calculated below:
                # (Note that in this example is multiplied
                # by a scale value, e.g. 100)
                self.fitness[i] = y * 100
            #########################################################

            print("fitness = ", i, " ", self.fitness[i])
            fitness_total = fitness_total + self.fitness[i]
        fitness_average = fitness_total / self.N
        i = 1
        while i <= self.N:
            sum_sqr = sum_sqr + pow(self.fitness[i] - fitness_average, 2)
            i = i + 1
        variance = sum_sqr / self.N
        if variance <= 1.0e-4:
            variance = 0.0
        # Best chromosome selection
        self.the_best_chrom = 0
        fitness_best = self.fitness[1]
        for i in range(1, self.popSize):
            if self.is_better_equal(self.fitness[i], fitness_best):
                fitness_best = self.fitness[i]
                self.the_best_chrom = i
        self.best_chrom[generation] = self.the_best_chrom
        # Statistical output                                   
        f = open("output.dat", "a")
        f.write(str(generation) + " " + str(fitness_average) + "\n")
        f.write("\n")
        f.close()
        print("Population size = ", self.popSize - 1)
        print("mean fitness = ", fitness_average)
        print("variance = ", variance, " Std. deviation = ", math.sqrt(variance))
        print("fitness best = ", fitness_best)
        print("chromosome best = ", self.best_chrom[generation])
        print("fitness sum = ", fitness_total)

    #########################################################
    # QUANTUM ROTATION GATE                                 #
    #########################################################
    def rotation(self):
        rot = np.empty([2, 2])
        # Lookup table of the rotation angle
        for i in range(1, self.popSize):
            for j in range(1, self.genomeLength):
                if self.is_worse(self.fitness[i], self.fitness[self.best_chrom[self.generation]]):
                    # if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:
                    if self.chromosome[i, j] == 0 and self.chromosome[self.best_chrom[self.generation], j] == 1:
                        # Define the rotation angle: delta_theta (e.g. 0.0785398163)
                        delta_theta = 0.0785398163
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        self.nqpv[i, j, 0] = (rot[0, 0] * self.qpv[i, j, 0]) + (rot[0, 1] * self.qpv[i, j, 1])
                        self.nqpv[i, j, 1] = (rot[1, 0] * self.qpv[i, j, 0]) + (rot[1, 1] * self.qpv[i, j, 1])
                        self.qpv[i, j, 0] = round(self.nqpv[i, j, 0], 2)
                        self.qpv[i, j, 1] = round(1 - self.nqpv[i, j, 0], 2)
                    if self.chromosome[i, j] == 1 and self.chromosome[self.best_chrom[self.generation], j] == 0:
                        # Define the rotation angle: delta_theta (e.g. -0.0785398163)
                        delta_theta = -0.0785398163
                        rot[0, 0] = math.cos(delta_theta)
                        rot[0, 1] = -math.sin(delta_theta)
                        rot[1, 0] = math.sin(delta_theta)
                        rot[1, 1] = math.cos(delta_theta)
                        self.nqpv[i, j, 0] = (rot[0, 0] * self.qpv[i, j, 0]) + (rot[0, 1] * self.qpv[i, j, 1])
                        self.nqpv[i, j, 1] = (rot[1, 0] * self.qpv[i, j, 0]) + (rot[1, 1] * self.qpv[i, j, 1])
                        self.qpv[i, j, 0] = round(self.nqpv[i, j, 0], 2)
                        self.qpv[i, j, 1] = round(1 - self.nqpv[i, j, 0], 2)
                # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

    #########################################################
    # X-PAULI QUANTUM MUTATION GATE                         #
    #########################################################
    # pop_mutation_rate: mutation rate in the population
    # mutation_rate: self.probability of a mutation of a bit 
    def mutation(self, pop_mutation_rate, mutation_rate):

        for i in range(1, self.popSize):
            up = np.random.random_integers(100)
            up = up / 100
            if up <= pop_mutation_rate:
                for j in range(1, self.genomeLength):
                    um = np.random.random_integers(100)
                    um = um / 100
                    if um <= mutation_rate:
                        self.nqpv[i, j, 0] = self.qpv[i, j, 1]
                        self.nqpv[i, j, 1] = self.qpv[i, j, 0]
                    else:
                        self.nqpv[i, j, 0] = self.qpv[i, j, 0]
                        self.nqpv[i, j, 1] = self.qpv[i, j, 1]
            else:
                for j in range(1, self.genomeLength):
                    self.nqpv[i, j, 0] = self.qpv[i, j, 0]
                    self.nqpv[i, j, 1] = self.qpv[i, j, 1]
        for i in range(1, self.popSize):
            for j in range(1, self.genomeLength):
                self.qpv[i, j, 0] = self.nqpv[i, j, 0]
                self.qpv[i, j, 1] = self.nqpv[i, j, 1]

    #########################################################
    # TOURNAMENT SELECTION OPERATOR                         #
    #########################################################
    def select_p_tournament(self):
        u1 = 0
        u2 = 0
        parent = 99
        while (u1 == 0 and u2 == 0):
            u1 = np.random.random_integers(self.popSize - 1)
            u2 = np.random.random_integers(self.popSize - 1)
            if self.is_better_equal(self.fitness[u1], self.fitness[u2]):
                parent = u1
            else:
                parent = u2
        return parent

    #########################################################
    # ONE-POINT CROSSOVER OPERATOR                          #
    #########################################################
    # crossover_rate: setup crossover rate
    def mating(self, crossover_rate):
        j = 0
        crossover_point = 0
        parent1 = self.select_p_tournament()
        parent2 = self.select_p_tournament()
        if random.random() <= crossover_rate:
            crossover_point = np.random.random_integers(self.genomeLength - 2)
        j = 1
        while (j <= self.genomeLength - 2):
            if j <= crossover_point:
                self.child1[parent1, j, 0] = round(self.qpv[parent1, j, 0], 2)
                self.child1[parent1, j, 1] = round(self.qpv[parent1, j, 1], 2)
                self.child2[parent2, j, 0] = round(self.qpv[parent2, j, 0], 2)
                self.child2[parent2, j, 1] = round(self.qpv[parent2, j, 1], 2)
            else:
                self.child1[parent1, j, 0] = round(self.qpv[parent2, j, 0], 2)
                self.child1[parent1, j, 1] = round(self.qpv[parent2, j, 1], 2)
                self.child2[parent2, j, 0] = round(self.qpv[parent1, j, 0], 2)
                self.child2[parent2, j, 1] = round(self.qpv[parent1, j, 1], 2)
            j = j + 1
        j = 1
        for j in range(1, self.genomeLength):
            self.qpv[parent1, j, 0] = self.child1[parent1, j, 0]
            self.qpv[parent1, j, 1] = self.child1[parent1, j, 1]
            self.qpv[parent2, j, 0] = self.child2[parent2, j, 0]
            self.qpv[parent2, j, 1] = self.child2[parent2, j, 1]

    def crossover(self, crossover_rate):
        c = 1
        while (c <= self.N):
            self.mating(crossover_rate)
            c = c + 1

    #########################################################
    # PERFORMANCE GRAPH                                     #
    #########################################################
    # Read the Docs in http://matplotlib.org/1.4.1/index.html
    def plot_Output(self):
        data = np.loadtxt('output.dat')
        # plot the first column as x, and second column as y
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y)
        plt.xlabel('Generation')
        plt.ylabel('fitness average')
        plt.xlim(0.0, 550.0)
        plt.show()

    #########################################################
    #                                                       #
    # MAIN PROGRAM                                          #
    #                                                       #
    #########################################################
    def Q_Hybrid(self):
        self.generation = 0
        print("============== GENERATION: ", self.generation, " =========================== ")
        print()
        self.Init_population()
        self.Show_population()
        self.Measure(0.5)
        self.fitness_evaluation(self.generation)
        while (self.generation < self.generation_max - 1):
            print("The best of generation [", self.generation, "] ", self.best_chrom[self.generation])
            print()
            print("============== GENERATION: ", self.generation + 1, " =========================== ")
            print()
            self.rotation()
            self.crossover(0.75)
            self.mutation(0.0, 0.001)
            self.generation = self.generation + 1
            self.Measure(0.5)
            self.fitness_evaluation(self.generation)

    #########################################################
    # ENTRY POINT                                           #
    #########################################################
    def run(self):
        print("""HYBRID GENETIC ALGORITHM""")
        input("Press Enter to continue...")
        self.Q_Hybrid()
        self.plot_Output()


if __name__ == '__main__':
    HGA().run()
