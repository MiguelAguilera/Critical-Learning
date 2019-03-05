import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mountain_car import MountainCarEnv


class ising:
  # Initialize the network
  def __init__(self, netsize, Nsensors=1, Nmotors=1):  # Create ising model

    self.size = netsize    #Network size
    self.Ssize = Nsensors  # Number of sensors
    self.Msize = Nmotors  # Number of sensors

    self.h = np.zeros(netsize)
    self.J = np.zeros((netsize, netsize))
    self.max_weights = 2

    self.randomize_state()

    self.env = MountainCarEnv()
    self.env.min_position = - np.pi / 2
    self.env.max_position =  np.pi / 6
    self.env.goal_position = np.pi / 6
    self.env.max_speed = 0.045
    self.observation = self.env.reset()

    self.Beta = 1.0
    self.defaultT = max(100, netsize * 20)

    self.Ssize1 = 0
    self.maxspeed = self.env.max_speed
    self.Update(-1)

  def get_state(self, mode='all'):
    if mode == 'all':
      return self.s
    elif mode == 'motors':
      return self.s[-self.Msize:]
    elif mode == 'sensors':
      return self.s[0:self.Ssize]
    elif mode == 'input':
      return self.sensors
    elif mode == 'non-sensors':
      return self.s[self.Ssize:]
    elif mode == 'hidden':
      return self.s[self.Ssize:-self.Msize]

  def get_state_index(self, mode='all'):
    return bool2int(0.5 * (self.get_state(mode) + 1))

  # Randomize the state of the network
  def randomize_state(self):
    self.s = np.random.randint(0, 2, self.size) * 2 - 1
    self.sensors = np.random.randint(0, 2, self.Ssize) * 2 - 1

  # Randomize the position of the agent
  def randomize_position(self):
    self.observation = self.env.reset()

  # Set random bias to sets of units of the system
  def random_fields(self, max_weights=None):
    if max_weights is None:
      max_weights = self.max_weights
    self.h[self.Ssize:] = max_weights * \
      (np.random.rand(self.size - self.Ssize) * 2 - 1)

  # Set random connections to sets of units of the system
  def random_wiring(self, max_weights=None):  # Set random values for h and J
    if max_weights is None:
      max_weights = self.max_weights
    for i in range(self.size):
      for j in np.arange(i + 1, self.size):
        if i < j and (i >= self.Ssize or j >= self.Ssize):
          self.J[i, j] = (np.random.rand(1) * 2 - 1) * self.max_weights

  # Update the position of the agent
  def Move(self):
    epsilon = np.finfo(float).eps
    self.previous_speed = self.observation[1]
    self.previous_vspeed = self.observation[1] * 3 * np.cos(3 * self.observation[0])
    action = int(np.digitize(
      np.sum(self.s[-self.Msize:]) / self.Msize, [-1 / 3+epsilon, 1 / 3, 1.1]))
    observation, reward, done, info = self.env.step(action)

    if self.env.state[0] >= self.env.max_position:  # Bounce when end of world is reached
      if self.observation[1] > 0:
        self.env.state = (self.env.max_position, 0)
      else:
        self.env.state = (self.env.max_position, self.observation[1])

    if self.env.state[0] <= self.env.min_position:  # Bounce when end of world is reached
      if self.observation[1] < 0:
        self.env.state = (self.env.min_position, 0)
      else:
        self.env.state = (self.env.min_position, self.observation[1])

    self.observation = self.env.state
    self.position = self.env.state[0]
    self.height = np.sin(3 * self.position)

    self.speed = self.env.state[1]

  # Transorm the sensor input into integer index
  def SensorIndex(self, x, xmax):
    return int(np.floor((x + xmax) / (2 * xmax + 10 *
                                    np.finfo(float).eps) * 2**self.Ssize))

  # Update the state of the sensor
  def UpdateSensors(self):
    self.speed_ind = self.SensorIndex(self.speed, self.maxspeed)
    self.sensors = 2 * bitfield(self.speed_ind, self.Ssize) - 1

  # Execute step of the Glauber algorithm to update the state of one unit
  def GlauberStep(self, i=None): 
    if i is None:
      i = np.random.randint(self.size)

    I = 0
    if i < self.Ssize:
      I = self.sensors[i]
    eDiff = 2 * self.s[i] * (self.h[i] + I +
                             np.dot(self.J[i, :] + self.J[:, i], self.s))
    if eDiff * self.Beta < np.log(1 / np.random.rand() - 1):    # Glauber
      self.s[i] = -self.s[i]

  # Update random unit of the agent
  def Update(self, i=None):
    if i is None:
      i = np.random.randint(-1, self.size)
    if i == -1:
      self.Move()
      self.UpdateSensors()
    else:
      self.GlauberStep(i)

  # Sequentially update state of all units of the agent in random order
  def SequentialUpdate(self):
    for i in np.random.permutation(range(-1, self.size)):
      self.Update(i)

  # Step of the learning algorith to ajust correlations to the critical regime
  def AdjustCorrelations(self, T=None):
    if T is None:
      T = self.defaultT

    self.m = np.zeros(self.size)
    self.c = np.zeros((self.size, self.size))
    self.C = np.zeros((self.size, self.size))

    # Main simulation loop:
    self.x = np.zeros(T)
    samples = []
    for t in range(T):

      self.SequentialUpdate()
      self.x[t] = self.position
      self.m += self.s
      for i in range(self.size):
        self.c[i, i + 1:] += self.s[i] * self.s[i + 1:]
    self.m /= T
    self.c /= T
    for i in range(self.size):
      self.C[i, i + 1:] = self.c[i, i + 1:] - self.m[i] * self.m[i + 1:]

#    c_ref = np.zeros((self.size, self.size))
#    for i in range(self.size):
#      inds = np.array([], int)
#      c = np.array([])
#      for j in range(self.size):
#        if not i == j:
#          inds = np.append(inds, [j])
#        if i < j:
#          c = np.append(c, [self.c[i, j]])
#        elif i > j:
#          c = np.append(c, [self.c[j, i]])
#      order = np.argsort(c)[::-1]
#      c_ref[i, inds[order]] = self.Cint[i, :]
#    self.c_ref = np.triu(c_ref + c_ref.T, 1)
#    self.c_ref *= 0.5

#    # Exclude sensor means
#    self.m[0:self.Ssize] = 0
#    self.m_ref[0:self.Ssize] = 0
#    # Exclude sensor, motor, and sensor-motor correlations
#    self.C[0:self.Ssize, 0:self.Ssize] = 0
##    self.C[-self.Msize:, -self.Msize:] = 0
#    self.C[0:self.Ssize, -self.Msize:] = 0
#    self.C_ref[0:self.Ssize, 0:self.Ssize] = 0
#    self.C_ref[-self.Msize:, -self.Msize:] = 0
#    self.C_ref[0:self.Ssize, -self.Msize:] = 0
    
    # Update weights
    dh = self.m_ref - self.m
    dJ = self.C_ref - self.C
    dh[0:self.Ssize]=0
    dJ[0:self.Ssize, 0:self.Ssize] = 0
    dJ[-self.Msize:, -self.Msize:] = 0
    dJ[0:self.Ssize, -self.Msize:] = 0
    
    
    # we make the matrix symmetric
#    i_lower = np.tril_indices(self.size, -1)
#    dJ[i_lower] = dJ.T[i_lower]
    
    return dh, dJ

  # Algorithm for poising an agent in a critical regime
  def CriticalLearning(self, Iterations, T=None):
    fitness=np.zeros(Iterations)
    u = 0.02
    count = 0
    dh, dJ = self.AdjustCorrelations(T)
    fit = max(np.max(np.abs(self.C_ref - self.C)), np.max(np.abs(self.m_ref - self.m)))
    x_min = np.min(self.x)
    x_max = np.max(self.x)
#    maxmin_range = (self.env.max_position + self.env.min_position) / 2
#    maxmin = (np.array([x_min, x_max]) - maxmin_range) / maxmin_range
#    print(count, fit, np.max(np.abs(self.J)))
    for it in range(Iterations):
      count += 1
      self.h += u * dh
      self.J += u * dJ
#      print(self.C)
#      print(self.C_ref)
#      print(self.J)
#      print(dJ)
#      input("Press Enter to continue...")
      if it % 10 == 0:
        self.randomize_state()
        self.randomize_position()
      Vmax = self.max_weights
      for i in range(self.size):
        if np.abs(self.h[i]) > Vmax:
          self.h[i] = Vmax * np.sign(self.h[i])
        for j in np.arange(i + 1, self.size):
          if np.abs(self.J[i, j]) > Vmax:
            self.J[i, j] = Vmax * np.sign(self.J[i, j])

      dh, dJ = self.AdjustCorrelations(T)

      fit = np.amax(np.abs(self.C_ref - self.C))
      fitness[it]=fit
      print(self.size,count,fit)
      if count % 1 == 0:
        mid = (self.env.max_position + self.env.min_position) / 2
#        print(self.size,count,fit,
#                                  np.mean(np.abs(self.J)),
#                                  np.max(np.abs(self.J)),
#                                  (np.min(self.x) - mid) / np.pi * 3,
#                                  (np.max(self.x) - mid) / np.pi * 3)
    return fitness

  def normalizePosition(self,pos):
    newPos = pos-self.env.min_position
    newPos = newPos/(self.env.max_position-self.env.min_position)
    newPos = newPos*2-1
    return newPos
  
  def simulate(self,T=1000,plot=False):
    self.env.reset()
    y = np.zeros(T)
    t = np.zeros(T)
    plt.figure(0)
    plt.xlim(-1,1)
    for x in range(T):
      self.SequentialUpdate()
      y[x]=self.normalizePosition(self.position)
      t[x]=x
    if(plot):
      plt.plot(y,t)
    return y
  
  def render(self,T=1000):
    self.env.reset()
    y = np.zeros(T)
    t = np.zeros(T)
    plt.figure(0)
    plt.xlim(-1,1)
    for x in range(T):
      self.SequentialUpdate()
      self.env._render()
      y[x]=self.normalizePosition(self.position)
      t[x]=x
      plt.scatter(y[x],x,color='k')
      plt.pause(0.05)
#      plt.show()
#    plt.figure(1)
#    plt.xlim(-1,1)
#    plt.plot(y,t)
    self.env.close()
    
# Transform bool array into positive integer
def bool2int(x):
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return int(y)

# Transform positive integer into bit array
def bitfield(n, size):
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)
