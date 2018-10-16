# stochastic simulation of AMPA receptor dynamics using the Gillespie algorithm
# Sanity Check: Plots different combinations of number of molecules and filling fraction of the system
# with all rates alpha, beta, delta, gamma unlike 0

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from copy import deepcopy
import time

###################################################
######## functions ################################
###################################################

def FF(alpha,beta,delta,gamma):
    """Calculates the short-term filling fraction"""
    
    F=1/(1+((beta*delta)/(alpha*gamma)))
    return(F)

def next_values(a0,a):
    """returns values for the next reaction like time difference and reaction"""
    
    # generate random numbers r1 and r2 from the normal distribution
    r1 = np.random.random(1)[0]
    r2 = np.random.random(1)[0]
    
    # calculate next time
    new_time_difference = (1/a0)*np.log(1/r1)
    
    # choose next reaction R_mu under the condition that
    # sum(a[i], i=0, mu-1) < r2*a0 <= sum(a[i], i=0, mu)
    mu = 0
    N = r2*a0 - a[mu] 
    
    while N > 0:
        mu += 1
        N = N - a[mu]
    
    return(new_time_difference, mu)
    
def calculate_hi(n,m):
    """calculates the hi with the help of binomial coeff and factorial() 
    since hi is defined as the product of total number of distinct 
    combinations of Ri reactant molecules (it's like Bin(X1,3)*3!)"""
    
    b=[0]*(n+1)
    b[0]=1
    for i in range(1,n+1):
        b[i]=1
        j=i-1
        while j>0:
            b[j]+=b[j-1]
            j-=1
    hi = b[m]*factorial(m)
    return(hi)

def show_gillespie(results, number_species, histo_range, name_file):
    """plots the results and shows a histogram of reaction"""
    store_time = results[0]
    store_number_molecules = results[1]
    f = plt.figure(figsize=(5,4))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    plt.rc('text', usetex=True)

    color_lines = ['coral','black','darkviolet','red','cyan','green','blue']
    labels = [r'$w_1$',r'$w_2$', r'$w_3$', r'$p$',r'$e_1$', r'$e_2$', r'$e_3$']
    for i in [3,6,2,5,1,4,0]:
        plt.plot(store_time, store_number_molecules[:,i],'k',color=color_lines[i],label=labels[i],linewidth=0.3)
    plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
    plt.ylabel('Molekuelanzahl', fontsize=12)
    plt.legend(loc=1, fontsize=11)
    #plt.title(r'$F=0.5$', fontsize=12)
    plt.show()
    f.savefig("long_X.pdf", bbox_inches='tight')

def show_gillespie_without_p(results, number_species, histo_range, name_file):
    """plots the results and shows a histogram of reaction"""
    
    store_time = results[0]
    store_number_molecules = results[1]
    f = plt.figure(figsize=(5,4))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    plt.rc('text', usetex=True)

    color_lines = ['coral','black','darkviolet','red','cyan','green','blue']
    labels = [r'$w_1$',r'$w_2$', r'$w_3$', r'$p$',r'$e_1$', r'$e_2$', r'$e_3$']
    for i in [3,6,2,5,1,4,0]:
        if i == 3:
            continue
        plt.plot(store_time, store_number_molecules[:,i],'k',color=color_lines[i],label=labels[i],linewidth=0.3)
    plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
    plt.ylabel('Molekuelanzahl', fontsize=12)
    plt.legend(loc=1, fontsize=11)
    #plt.title(r'$F=0.5$', fontsize=12)
    plt.show()
    f.savefig("long_without_p.pdf", bbox_inches='tight')
    
def reactions_stoch(reactions):
    """gets a string of several reactions and outputs the stoichiometry array
    of substrates and products
    
    the form of the string is like '2X1+X2->X3,X1->0,X3+4X2->12X1'
    to define N species please use X1, X2, X3,..., XN-1, XN 
    only use symbols like '->' and '+', dont use spaces
    """
    # important variables
    substrates = [] # string reactions is splitted into substrates and products
    products = []
    sub_without_plus = [] # each subtrate is separated by '+'
    prod_without_plus = []
    sub_number_reaction = [] # list with items like [number of species, name of species]
    prod_number_reaction = []
    total_number_species = 0 # number of all species to create array
    
    # remove symobls like '+', '->', ',' and split it into substrates and products 
    one_reaction = reactions.split(",")
    for i in one_reaction:
        s_p = i.split("->")
        substrates.append(s_p[0])
        products.append(s_p[1])

    for i in substrates:
        sub_without_plus.append(i.split("+"))
    for i in products:
        prod_without_plus.append(i.split("+"))
    
    # split each item into [number of species, name of species]
    for i in sub_without_plus:
        sub_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                sub_one_reaction.append(number_reaction)
        if len(sub_one_reaction) >= 1:
            sub_number_reaction.append(sub_one_reaction)
    
    for i in prod_without_plus:
        prod_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                prod_one_reaction.append(number_reaction)
        if len(prod_one_reaction) >= 1:
            prod_number_reaction.append(prod_one_reaction)
    
    # create arrays for the stoichiometry of substrates and products
    sub_stoch = np.zeros((len(one_reaction), total_number_species), int)
    prod_stoch = np.zeros((len(one_reaction), total_number_species), int)
    
    # fill the arrays with the number of species
    for reaction in range(len(sub_number_reaction)):
        for species in sub_number_reaction[reaction]:
            sub_stoch[reaction][species[1]-1] = int(-species[0])
    
    for reaction in range(len(prod_number_reaction)):
        for species in prod_number_reaction[reaction]:
            prod_stoch[reaction][species[1]-1] = int(species[0])
    return(sub_stoch, prod_stoch)

###################################################
######## Gillespie algorithm ######################
###################################################
    
def gillespie_algo(number_actual_synapses, init, rates, sub_stoch, prod_stoch, tmax, n_max):
    """solver for stochastic equations, outputs the time, number of molecules and
    reaction
    
    [init = initial conditions of the reactants in an array (X1,X2,...)
    rates = ci parameters in an array
    sub_stoch, prod_stoch = stochiometry of substrates and products
    tmax = maximum time
    n_max = maximum number of reactions]
    """
    
    # ****************************   
    # step 0: input rate values, initial values and initialise time and reactions counter
    # ****************************   
    stoch = sub_stoch + prod_stoch
    number_reactions = np.shape(stoch)[0] # number of reactions
    number_species = np.shape(stoch)[1] # number of species

    # initialise current time, current species variables, time and reaction counters
    current_time = 0
    current_species = init
    #print("init:", init)
    n_counter = 0
    
    # initialise variables to store time and molecule numbers
    store_time = np.zeros((n_max, 1))
    store_number_molecules = np.zeros((n_max, number_species))
    store_reaction = np.zeros((n_max, 1))

    # store current time, state of system, filling fraction
    store_time[n_counter] = current_time
    store_number_molecules[n_counter,:] = current_species 
    store_constant = [[],[],[],[]]    
    store_filling_fraction_av = np.array([0.0,0.0,0.0])
    cv_start = 0
    
    while (current_time < tmax):
        
        # ****************************   
        # step 1: calculate ai and a0
        # ****************************   
        
        if current_time >= 1 and cv_start == 0:
            cv_start = deepcopy(n_counter)

        a = np.ones((number_reactions,1))


        for i in range(number_reactions):
            hi = 1  # h1 is defined as the product of total number of distinct 
                    # combinations of Ri reactant molecules 
            for j in range(number_species):

                # check the reactant is involved in this reaction
                if sub_stoch[i,j] == 0:
                    continue
                else:
                    # check the reactant has molecules available
                    if current_species[j] <= 0: 
                        # text = "Loop " + str(n_counter) + ": Reactant X" + str(j+1) + " has 0 molecules"
                        # print(text)
                        hi = 0
                        continue
                    else:
                        hi *= calculate_hi(current_species[j],np.absolute(sub_stoch[i,j]))
                        
            a[i] = hi*rates[i]  # ai = hi * ci

        a0 = sum(a)

        # ****************************   
        # step 3: update and store system
        # ****************************   
        
        # choose the next time difference and reaction
        tr = next_values(a0,a)
        new_time_difference = tr[0]
        next_r = tr[1]

        # update time, number species, counter
        current_time += new_time_difference 
        current_species += np.transpose(stoch[next_r,:])
        n_counter += 1

        # store current system
        store_time[n_counter] = current_time
        store_number_molecules[n_counter,:] = current_species 
        store_reaction[n_counter] = next_r
        
        # for checking total size of receptors and slots
        store_constant[0].append\
        (current_species[0]+current_species[1]+current_species[2]+current_species[3])
        for i in range(1,4):
            store_constant[i].append(current_species[i-1] + current_species[i+3])

        store_filling_fraction_av += new_time_difference*current_species[:3].astype(np.float64)
                
    # store final output
    store_time = store_time[:n_counter]
    store_number_molecules = store_number_molecules[:n_counter,:]
    store_reaction = store_reaction[:n_counter]
    
    # calculate final filling fraction
    store_filling_fraction_av /= (current_time*number_actual_synapses)
    
    # calculate coefficient of variation
    mol_cv = store_number_molecules[cv_start:,:4]
    average = sum(mol_cv)
    average /= len(mol_cv) # [average1, average2,...]
    
    coefficient_variation = np.sqrt(sum((mol_cv - average)**2)/len(mol_cv))*100/average
    
    print("counter n:", n_counter)
    return(store_time, store_number_molecules, store_reaction, store_constant, np.array(store_filling_fraction_av), np.array(coefficient_variation))

###################################################
######## Set start values #########################
###################################################

# define the stochiometry of the substrates and products separatelly
sub_stoch, prod_stoch = reactions_stoch("X4+X5->X1,X4+X6->X2,X4+X7->X3,X1->X4+X5,X2->X4+X6,X3->X4+X7,X4->0X4,0X4->X4")

# define the initial conditions of the reactants
F = 0.5
slots_s1 = 40
slots_s2 = 60
slots_s3 = 80
s = slots_s1 + slots_s2 + slots_s3
beta = 60/43
alpha = beta/(2.67*s*(1-F))
delta = 1/14
gamma = delta*F*s*2.67
p = round(gamma/delta)
W = p/2.67
tmax = 5
n_max = 6000
name_file = "Results"
name_sim = "With gamma delta"
number_sim = "1"
notes = "Notes: -"
sim_round = 1
times_sim_av = 1 # number of repeated simulations for average

###################################################
######## Run the main program #####################
###################################################


rates = np.array([alpha,alpha,alpha,beta,beta,beta,delta,gamma])
init1 = np.array([0,0,0,p,slots_s1,slots_s2,slots_s3])
number_actual_synapses = deepcopy(init1[4:])
slots_ff_percent = np.round(FF(alpha,beta,delta,gamma),2)

number_empty_slots = sum(init1[4:])
init1[0] = ((init1[4]/number_empty_slots)*W)//1
init1[1] = ((init1[5]/number_empty_slots)*W)//1
init1[2] = ((init1[6]/number_empty_slots)*W)//1
init1[4] -= init1[0]
init1[5] -= init1[1]
init1[6] -= init1[2]

# create text file with important information
name_sim += " " + number_sim
data_in = open(name_file,"a+")
data_in.write(name_sim+"\n")
print(name_sim+"\n")
data_in.write(time.strftime("%d.%m.%Y %H:%M:%S")+"\n")
print(time.strftime("%d.%m.%Y %H:%M:%S"))
data_in.write(notes+"\n")
print(notes+"\n")
data_in.write("\n")
print("")
data_in.write("3 Synapses with "+str(number_actual_synapses)+" slots\n")
print("3 Synapses with", number_actual_synapses, "slots")
data_in.write("p = "+str(int(p))+"\n")
print("p =", int(p))
data_in.write("alpha = "+str(rates[0])+", beta = "+str(rates[3])+"\n")
print("alpha =",str(rates[0])+", beta =",rates[3])
data_in.write("delta = "+str(rates[6])+", gamma = "+str(rates[7])+"\n")
print("delta = "+str(rates[6])+", gamma = "+str(rates[7]))
data_in.write("Chosen Filling Fraction: "+str(F)+" \n")
print("Chosen Filling Fraction: "+str(F))
data_in.write("Theoretical Filling Fraction: "+str(slots_ff_percent)+" \n")
print("Theoretical Filling Fraction: "+str(slots_ff_percent))
data_in.write("------------------ ")
data_in.write("\n")
print("------------------")
data_in.write("Results: \n")
print("Results: \n")

# check the result for filling fraction and coeff var
for r in range(1,sim_round+1):
    ff_av = np.array([0.0,0.0,0.0])
    cv = np.array([0.0,0.0,0.0,0.0])
    
    counter = 0
    counter_print = 0
    while True:
        init = deepcopy(init1)
        if counter == times_sim_av:
            break
        results = gillespie_algo(number_actual_synapses, init, rates, sub_stoch, prod_stoch, tmax, n_max)
        store_filling_fraction_av, coefficient_variation = results[4:]
        # check whether there is a nan in coefficient_variation
        
        nan_there = 0
        for i in coefficient_variation:
            if np.isnan(i):
                nan_there = 1
        
        ff_av += store_filling_fraction_av
        
        print(coefficient_variation)
        
        if nan_there == 1:
            continue
        cv += coefficient_variation
        counter += 1
        counter_print += 1
        print("counter print: ", counter_print)
        if counter_print <= 4:
            print("")
            print("Simulation "+str(counter_print))
            print("Number of molecules of species:")
            show_gillespie(results[:3], len(init), [-0.5,0.5,1.5], name_sim+" X_i "+str(counter_print))
            print("")
            print("Number of molecules of species without pool p:")
            show_gillespie_without_p(results[:3], len(init), [-0.5,0.5,1.5], name_sim+" X_i without p "+str(counter_print))
            print("")
        
        store_time = results[0]
        store_constant = results[3]
        f = plt.figure(figsize=(5,4))
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 12}
        plt.rc('font', **font)
        plt.rc('font', serif='Times New Roman') 
        plt.rc('text', usetex=True)

        plt.plot(store_time, store_constant[0], 'k',color='slategray', linewidth=0.3, label = r'$R$')
        plt.plot(store_time, store_constant[3], 'k',color='darkviolet', linewidth=0.3, label = r'$s_3$')
        plt.plot(store_time, store_constant[2], 'k',color='black', linewidth=0.3, label = r'$s_2$')
        plt.plot(store_time, store_constant[1], 'k',color='coral', linewidth=0.3, label = r'$s_1$')
        plt.legend(loc=1, fontsize=11)
        plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
        plt.ylabel('Molekuelanzahl', fontsize=12)
        plt.show()
        f.savefig("long_constant.pdf")
        
    ff_av /= times_sim_av
    cv /= times_sim_av

    data_in.write("Number of repeated simulations: "+str(counter)+"\n")
    print("Number of repeated simulations: "+str(counter))
    data_in.write("Values for synapse 1,2,3:\n")
    print("Values for synapse 1,2,3:")
    data_in.write("F average: "+str(list(np.round(ff_av, 2)))+"\n")
    print("F average:", list(np.round(ff_av, 2)))
    data_in.write("Coefficient of Variation: "+str(list(np.round(cv[:3], 2)))+"\n")
    print("Coefficient of Variation:", list(np.round(cv[:3], 2)))
    data_in.write("Coefficient of Variation of pool: "+str(np.round(cv[3], 2))+"\n")
    print("Coefficient of Variation of pool:", np.round(cv[3], 2))
    print("")
