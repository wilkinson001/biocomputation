##imports
import random
import pandas as pd
import numpy as np
import csv
##class definitions
class rule:
    def __init__(self, conditions, output):
        self.conditions = conditions or []
        self.output = output

class individual:
    def __init__(self, gene, fitness):
        self.genes = gene or []
        self.fitness = fitness

##method definitions
def gene_to_rule(individual):
    k=0
    rulebase=[]
    for i in range(0,rule_no):
        conditions = []
        for j in range(0,cond_len):
            conditions.append(individual.genes[k])
            k+=1
        output = individual.genes[k]
        k+=1
        rulebase.append(rule(conditions,output))
    return rulebase

def fitness_function(rulebase, df_rules, individual):
    for b in range(0, len(df_rules)):
        for a in range(0,rule_no):
            n=0
            for c in range(0,cond_len):
                if rulebase[a].conditions[c] == df_rules[b].conditions[c] or rulebase[a].conditions[c]==2:
                    n+=1
            if n==cond_len:
                if rulebase[a].output == df_rules[b].output:
                    individual.fitness+=1
                break
            
    return individual
#Change below to adjust for dataset 1 and 2
pop_no = 50
gene_len = 60
rule_no = 10
cond_len = 5
generations = 50
##main loop
population=[]
pop_fitness = 0
rulebase=[]
prob_mut = 1/81
mod_no=7

options = [pop_no,gene_len,rule_no,cond_len,generations,prob_mut]

csvoutput = open('dataset1_scratch.csv', 'a', newline='')
writer = csv.writer(csvoutput)

writer.writerow(['pop_no','gene_len','rule_no','cond_len','generations','prob_mut'])
writer.writerow(options)
writer.writerow(['generation','mean_fitness','best_fitness'])
#read file
#adjust file name for dataset 1 or 2
df = pd.read_csv('data1.txt', skiprows=1, header=None, delim_whitespace=True, dtype={0: np.str, 1: np.int})
df_rules=[]
for d in range(0,len(df)):
    #map df data to new rule base for comparison
    df_rule = list(map(int,df[0][d]))
    indv = rule(df_rule, df[1][d])
    df_rules.append(indv)

#print("df_rules")
for x in range(0,len(df_rules)):
    print(df_rules[x].conditions, " " , df_rules[x].output)
##generate first population
for x in range(0,pop_no):
    gene=[]
    for y in range(0,gene_len):
        if (y+1)%mod_no==0:
        #if y in mod_list:
            gene.append(random.randint(0,1))
        else:
            gene.append(random.randint(0,2))        
    ind=individual(gene,0)
    rulebase = (gene_to_rule(ind))
    ind = fitness_function(rulebase,df_rules,ind)
    population.append(ind)

print("rulebase")
for z in range(0,rule_no):
    print(rulebase[z].conditions, " ", rulebase[z].output)

max_fit=0
max_mean=0
overall_best_indv=population[0]
##loop for number of generations
for g in range(0,generations):

    ##code to find best fit in pop and mean fit in pop
    
    mean_fit=0
    best_fit = population[0].fitness
    worst_fit = population[0].fitness
    #for ii in range(0,pop_no):
    for f in range(0,pop_no):
        mean_fit+=population[f].fitness
        if population[f].fitness>=best_fit:
            best_fit = population[f].fitness
            best_indv = population[f]
        if population[f].fitness<=worst_fit:
            worst_fit = population[f].fitness
            worst_indv_location = f
        
    mean_fit-=population[worst_indv_location].fitness
    population[worst_indv_location] = best_indv
    mean_fit+=population[worst_indv_location].fitness
    if best_indv.fitness>overall_best_indv.fitness:
        overall_best_indv = best_indv
    if best_fit>max_fit:
        max_fit = best_fit
    if mean_fit>max_mean:
        max_mean = mean_fit
    mean_fit = mean_fit/pop_no
    print("best = ",best_fit)
    print("mean = ",mean_fit)
    writer.writerow([g,mean_fit,best_fit])
    ##loop to select mating pool
    #print("pop_fitness = ",pop_fitness)
    offspring = []
    off_fitness = 0
    pop_fitness = 0
    for z in range(0,pop_no):
        parent1 = random.randint(0,pop_no-1)
        parent2 = random.randint(0,pop_no-1)
        if population[parent1].fitness >= population[parent2].fitness:
            offspring.append(population[parent1])
            off_fitness += population[parent1].fitness
        else:
            offspring.append(population[parent2])
            off_fitness += population[parent2].fitness
            
    #print("off_fitness = ",off_fitness)
    ##shuffle pool
    random.shuffle(offspring)
##perform crossover and mutation
    ##crossover of consecutive parents (probability) else copy
    for x in range(0,int(pop_no/2)):
        ##setup individuals to work on
        p1 = offspring[x]
        p2 = offspring[x+1]
        g1=p1.genes
        g2=p2.genes
        ##perform crossover
        cross_point =  random.randint(0,gene_len)
        ng1 = g1[:cross_point]
        ng2 = g2[:cross_point]
        ng1.extend(g2[cross_point:])
        ng2.extend(g1[cross_point:])
        ##perform mutation
        for w in range(0, gene_len):
            if (w+1)%mod_no==0:
            #if w in mod_list:
                if random.randint(0,1000)/1000<prob_mut:
                    if ng1[w]==0:
                        ng1[w]=1
                    else:
                        ng1[w]=0
                if random.randint(0,1000)/1000<prob_mut:
                    if ng2[w]==0:
                        ng2[w]=1
                    else:
                        ng2[w]=0
            else:
                if random.randint(0,1000)/1000<prob_mut:
                    if ng1[w]==0:
                        if random.randint(0,100)<50:
                            ng1[w]=1
                        else:
                            ng1[w]=2
                    elif ng1[w]==1:
                        if random.randint(0,100)<50:
                            ng1[w]=0
                        else:
                            ng1[w]=2
                    else:
                        if random.randint(0,100)<50:
                            ng1[w]=0
                        else:
                            ng1[w]=1
                if random.randint(0,1000)/1000<prob_mut:
                    if ng2[w]==0:
                        if random.randint(0,100)<50:
                            ng2[w]=1
                        else:
                            ng2[w]=2
                    elif ng2[w]==1:
                        if random.randint(0,100)<50:
                            ng2[w]=0
                        else:
                            ng2[w]=2
                    else:
                        if random.randint(0,100)<50:
                            ng2[w]=0
                        else:
                            ng2[w]=1                
        ##work out fitness of new offspring
        i1 = individual(ng1,0)
        i2 = individual(ng2,0)
        
        i1 = fitness_function(gene_to_rule(i1),df_rules,i1)
        i2 = fitness_function(gene_to_rule(i2),df_rules,i2)
        pop_fitness+=i1.fitness
        pop_fitness+=i2.fitness
        ##replace pop with offspring
        population[x] = i1
        population[x+1] = i2
        x+=1
        
        
#for p in range(0,pop_no):
#    rulebase = gene_to_rule(population[p])
#    print("rule ", p)
#    print(population[p].genes," ", population[p].fitness)
#    for z in range(0,rule_no):
#        if(rulebase[z].output==2):
#            print(z," ", rulebase[z].conditions, " ", rulebase[z].output)
        #print(population[p].genes," ", population[p].fitness)
writer.writerow(['example genes'])
for pp in range(0,pop_no):
    if population[pp].fitness==64:
        writer.writerow([population[pp].genes])
print("max_fit = ",max_fit)
print("max_mean = ",max_mean)
print( overall_best_indv.fitness," ",overall_best_indv.genes)
zz = gene_to_rule(overall_best_indv)
for z in range(0,rule_no):
    print(z," ", zz[z].conditions, " ", zz[z].output)
    


csvoutput.close()