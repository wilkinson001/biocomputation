##imports
import random
import pandas as pd
import numpy as np
import csv
##class definitions
class rule:
    def __init__(self, conditions, output):
        self.conditions = conditions
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

def fitness_function(rulebase, rules, individual):
    #individual.fitness=0
    for b in range(0, len(rules)-1):
        for a in range(0,rule_no):
            n=0
            for c in range(0,cond_len-1):
                #print(rulebase[a].conditions[c],' ',df_rules[b].conditions[int(c/2)],' ',rulebase[a].conditions[c+1])
                if(c%2==1):
                    continue
                if rulebase[a].conditions[c] <= rules[b].conditions[int(c/2)] <=rulebase[a].conditions[c+1]:
                    n+=1
                if rulebase[a].conditions[c] >= rules[b].conditions[int(c/2)]>=rulebase[a].conditions[c+1]:
                    n+=1
            if n==cond_len/2:
                #print(rulebase[a].output," ", df_rules[b].output)
                if rulebase[a].output == rules[b].output:
                    individual.fitness+=1
                    #print('fit individual ', individual.genes)
                break
            
    return individual

pop_no = 100
gene_len = 130
rule_no = 10
cond_len = 12
generations = 600
##main loop
population=[]
pop_fitness = 0
rulebase=[]
prob_mut = 1/65
step_size=0.12#step size for mutation
mod_no=13#now for whether it is an output


options = [pop_no,gene_len,rule_no,cond_len,generations,prob_mut]

csvoutput = open('fp4.csv', 'a', newline='')
writer = csv.writer(csvoutput)

writer.writerow(['pop_no','gene_len','rule_no','cond_len','generations','prob_mut'])
writer.writerow(options)
writer.writerow(['generation','mean_fitness','best_fitness'])
#read training file
df = pd.read_csv('data3_training.txt', skiprows=1, header=None, delim_whitespace=True, dtype={0: np.float32, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.float32, 5: np.float32, 6: np.int})
df_rules=[]
for d in range(0,len(df)):
    #print(df)
    #map df data to new rule base for comparison
    df_rule=[]
    d_rule=[]
    
    for e in range (0,len(df.columns)-1):
        #print(df[df.columns[e]][d])
        d_rule.append(df[df.columns[e]][d])
    
    indv = rule(d_rule, df[6][d])#changed to account for split up conditions
    df_rules.append(indv)
#read full file
df_full = pd.read_csv('data3.txt', skiprows=1, header=None, delim_whitespace=True, dtype={0: np.float32, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.float32, 5: np.float32, 6: np.int})
df_rules_full=[]
for d in range(0,len(df_full)):
    #print(df)
    #map df data to new rule base for comparison
    df_rule_full=[]
    d_rule_full=[]
    
    for e in range (0,len(df_full.columns)-1):
        #print(df[df.columns[e]][d])
        d_rule_full.append(df_full[df_full.columns[e]][d])
    
    indv = rule(d_rule_full, df_full[6][d])
    df_rules_full.append(indv)


print("df_rules")
for x in range(0,len(df_rules)):
     print(df_rules[x].conditions, " " , df_rules[x].output)
#generate first population
pop_fit=0

for x in range(0,pop_no):#changed to go from 1 to account for hardcoded first individual.
    gene=[]
    for y in range(0,gene_len):
        if (y+1)%mod_no==0:#needed for output of individual
            gene.append(random.randint(0,1))
        else:
            gene.append(random.random())        
    ind=individual(gene,0)
    rulebase = (gene_to_rule(ind))
    ind = fitness_function(rulebase,df_rules,ind)
    population.append(ind)
    pop_fit+=ind.fitness

print("rulebase")
for z in range(0,rule_no):
    print(rulebase[z].conditions, " ", rulebase[z].output)


max_fit=0
max_mean=0
overall_best_indv=population[0]
##loop for number of generations
for g in range(0,generations):
    print("Generation: ",g)
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
        cross_point =  random.randint(0,gene_len-1)
        ng1 = g1[:cross_point]
        ng2 = g2[:cross_point]
        ng1.append(g2[cross_point])
        ng2.append(g1[cross_point])
        ng1.extend(g1[cross_point+1:])
        ng2.extend(g2[cross_point+1:])
        ##perform mutation
        for w in range(0, gene_len):
            if (w+1)%mod_no==0:
                #keep mutation the same for the ouput bit
                if random.random()<prob_mut:
                    if ng1[w]==0:
                        ng1[w]=1
                    else:
                        ng1[w]=0
                if random.random()<prob_mut:
                    if ng2[w]==0:
                        ng2[w]=1
                    else:
                        ng2[w]=0
            else: #change mutation to step rather than flip
                if random.random()<prob_mut:
                    if random.random()<=0.5:
                        if(abs(ng1[w]+random.uniform(0,step_size))>1):
                            ng1[w]=abs(ng1[w]-random.uniform(0,step_size))
                        else:
                            ng1[w]=abs(ng1[w]+random.uniform(0,step_size))
                    else:
                        if(abs(ng1[w]-random.uniform(0,step_size))<0):
                            ng1[w]=abs(ng1[w]+random.uniform(0,step_size))
                        else:
                            ng1[w]=abs(ng1[w]-random.uniform(0,step_size))
                if random.random()<prob_mut:
                    if random.random()<=0.5:
                        if(abs(ng2[w]+random.uniform(0,step_size))>1):
                            ng2[w]=abs(ng2[w]-random.uniform(0,step_size))
                        else:
                            ng2[w]=abs(ng2[w]+random.uniform(0,step_size))
                    else:
                        if(abs(ng2[w]-random.uniform(0,step_size))<0):
                            ng2[w]=abs(ng2[w]+random.uniform(0,step_size))
                        else:
                            ng2[w]=abs(ng2[w]-random.uniform(0,step_size))
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
        
        
for p in range(0,pop_no):
    population[p].fitness=0
    ind = fitness_function(gene_to_rule(population[p]),df_rules_full,population[p])    
    population[p]=ind
    #print("rule ", p)
    print(ind.genes," ", ind.fitness)

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

mean_fit = mean_fit/pop_no
writer.writerow(["max fitness on full set","mean fitness on full set"])
writer.writerow([best_fit, mean_fit])
        #print(population[p].genes," ", population[p].fitness)
print("max_fit = ",best_fit)
print("max_mean = ",mean_fit)
#print( overall_best_indv.fitness," ",overall_best_indv.genes)
#zz = gene_to_rule(overall_best_indv)
#for z in range(0,rule_no):
#    print(z," ", zz[z].conditions, " ", zz[z].output)

csvoutput.close()