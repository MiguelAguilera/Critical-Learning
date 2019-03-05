#!/usr/bin/env python


from embodied_ising import ising
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import ising_correlations as ic
from copy import copy
import numpy.random as rnd

plt.rc('text', usetex=True)
font = {'family':'serif','size':15, 'serif': ['computer modern roman']}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':13})

###############################
"""
    Variables
"""
########## Network ##########
size = 4
Nsensors = 1
Nmotors =1

########## Lattice reconfiguration ##########
L=100 # Lattice size
corrs=1 # number of lattice reconfigurations 

########## Microbial Genetic Algorithm ##########
P=1 # Population size
TourBatchSize = 10 # number of microbial tournament executions per repetition

########## Critical Cognitive Learning ##########
T =500 # number of samples/iteration
Iterations = 100
repetitions = 10


###############################
"""
    Method to calculate the level of adjustment between
    real and reference correlations
    Parameters:
      I: Ising model with actual and reference correlations
    Returns:
      fit: the fitness
      i: the index of the element of the network with the worst correlations
"""
def calculateCorrelationFitness(I):
  fit = np.amax(np.abs(I.C_ref - I.C))
  diffC = np.abs(I.C_ref - I.C)
  #looking for the element with the worst correlations
  s=np.sum(diffC,axis=0)
  i=s.argmax()
  return fit,i
###############################
"""
    Method to find a better lattice configuration in order to approximate 
    the reference correlations to those generated by a given Ising model
"""
def adjustRefCorrelations(I,mode='all'):
  fit_old,index = calculateCorrelationFitness(I)
  I_new = copy(I)
#  I_new.m_ref = ic.random_means(I.size)
  I_new.m_ref = np.zeros(I.size)
  I_new.pos = ic.move_one_position(I_new,mode=mode)
  I_new.C_ref = ic.ising_correlations(I_new.pos, I_new.m_ref)
  fit_new,index_new = calculateCorrelationFitness(I_new)
  diffFit=fit_new-fit_old
#  print("diffFit: "+str(diffFit))
  if(diffFit<0):# the new fit is better
#    print("corr fit: "+str(fit_new))
    return I_new 
  else: 
#    print("corr fit: "+str(fit_old))
    return I
  

###############################
"""
    Microbial genetic algorithm
    used to find the best agent
"""
def microbial_tournament(Population,fit):
  
  #Recombination and mutation rates
  REC=0.5
  MUT=0.1
  
  #Seleccionamos dos células aleatoriamente para el torneo
  i=np.random.randint(len(Population))
  j=np.random.randint(len(Population))
  while j==i:
    j=np.random.randint(len(Population))

  # Tournament
  # Fitness evaluation for each configuration
  fit[i]=calculateCognitiveFitness(Population[i])
  fit[j]=calculateCognitiveFitness(Population[j])
  
  if fit[j]>fit[i]:  #we change the order to maintain j as looser
      i1=i
      i=j
      j=i1
  I=Population[i]
  J=Population[j]
  #Recombination and mutation
  for ind in range(I.size**2):
    r1,r2 = np.unravel_index(ind,(I.size,I.size))
    if np.random.rand(1)<REC:
        J.J[r1,r2]=I.J[r1,r2]
    if  np.random.rand(1)<MUT:
        J.J[r1,r2]*=rnd.uniform(-2,2)

###############################
"""
    Method to calculate the performance of the ising model 
    in terms of the cognitive task
    
    Task: maximize the time in the right hill
"""
def calculateCognitiveFitness(I,nSamples = 1000,plot=False):
  data = I.simulate(nSamples,plot)
  fit = np.mean(data>0.8)
  return fit
###############################
"""
    Initialize population
"""

Population = []
fitness=np.zeros(P)

for i in range(P):
  I = ising(size, Nsensors, Nmotors)
#  I.m_ref = ic.random_means(size)
  I.m_ref = np.zeros(size)
  I.pos = ic.random_positions(L, L, size)
  I.C_ref = ic.ising_correlations(I.pos, I.m_ref)
  Population.append(I)
#  fitness[i]=calculateCognitiveFitness(I)


###############################
"""
    Update correlation (Population level)
"""
def updateCorrelations(Population,fitness):
  reconfErrors=np.zeros(corrs)
  criticalErrors = np.zeros(Iterations)
  for p in range(P):
    I=Population[p]
    criticalErrors = I.CriticalLearning(Iterations,T)
    printRefPos(I.pos)
    printDistanceMatrix(I.size,I.C_ref)
    printDistanceMatrix(I.size,I.C)
    input("pulse tecla...")
#    print("critical errors")
#    print(criticalErrors)
    for i in range(corrs):
      I=adjustRefCorrelations(I,mode='hidden')
      fit,index=calculateCorrelationFitness(I)
      reconfErrors[i]=fit
    Population[p]=I
#  best = bestFitness.argmax()
  return reconfErrors,criticalErrors
    
###############################
"""
    #Update cognitive fitness (Population level)
"""
def updateCognitiveFitness(Population,fitness):
  
  for p in range(P):
    I=Population[p]
    fitness[p]=calculateCognitiveFitness(I)
    
###############################
"""
    
"""
def printDistanceMatrix(size,C):
  
  distanceMatrix = np.zeros((size,size))
  for i in range(size):
    for j in range(i+1,size):
      distanceMatrix[i,j]=np.floor(np.abs(C[i,j]/0.9)**-4)
  print(distanceMatrix)
  
###############################
"""
    
"""
def printRefPos(pos):
  distances=np.zeros((len(pos),len(pos)))
  for i in range(len(pos)):
    for j in range(i+1,len(pos)):
      distances[i,j]=np.abs(pos[i,0]-pos[j,0])+np.abs(pos[i,1]-pos[j,1])
  print(distances)
    
###############################
"""
    Training algorithm
"""
cogFitness=np.zeros(repetitions*TourBatchSize)
reconfErrors=np.zeros(repetitions*corrs)
criticalErrors = np.zeros(repetitions*Iterations)
for rep in range(repetitions):
  reconfE,criticalE =  updateCorrelations(Population,fitness)
  reconfErrors[rep*corrs:(rep+1)*corrs] = reconfE
  criticalErrors[rep*Iterations:(rep+1)*Iterations] = criticalE
#  plt.scatter(rep,corrFitness[rep],color='k')
#    plt.pause(0.05)
#  updateCognitiveFitness(Population,fitness)
#  for i in range(TourBatchSize):
#    microbial_tournament(Population,fitness)
#    best = fitness.argmax()
#    f=fitness[best]
#    x=rep*TourBatchSize+i
#    cogFitness[x]=f
#    print("Best fitness: "+str(f))

plt.plot(range(len(reconfErrors)),reconfErrors)
plt.xlabel("Reconfiguration index")
plt.ylabel("Error")
plt.title('Correlation errors during lattice reconfiguration')
xcoords = []
for i in range(repetitions):
  xcoords.append(i*corrs)
for xc in xcoords:
    plt.axvline(x=xc,color='gray',linestyle='--',linewidth=0.5)
plt.savefig("./reconfigurationErrors"+str(size)+".pdf")

plt.figure()
plt.plot(range(len(criticalErrors)),criticalErrors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title('Correlation errors during Critical Learning')
xcoords = []
for i in range(repetitions):
  xcoords.append(i*Iterations)
for xc in xcoords:
    plt.axvline(x=xc,color='gray',linestyle='--',linewidth=0.5)
plt.savefig("./LearningErrors"+str(size)+".pdf")
#plt.plot(cogFitness)
#plt.title('Cognitive Fitness')

#best = fitness.argmax()
#print("Best fitness: "+str(fitness[best]))
#Ibest = Population[best]
#Ibest.render()
      
      
      
      
#    j=rep*corrs+i
#    y[j]=fit
#    plt.scatter(j,y[j],color='k')
#    plt.pause(0.05)

#plt.plot(y)
#  print("Calculating Performance...")
#  fit = calculateCognitiveFitness(I,1000,False)
#  print(fit)
    

###############################
"""
    Save model
"""
#  filename = 'files/network-size_' + str(size) + '-sensors_' + str(Nsensors) + '-motors_' + str(
#      Nmotors) + '-T_' + str(T) + '-Iterations_' + str(Iterations) + '-ind_' + str(rep) + '.npz'
#np.savez(filename, J=I.J, h=I.h, m1=I.m_ref, Cint=I.C_ref)