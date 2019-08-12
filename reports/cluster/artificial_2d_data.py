import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

# Number of features
d = 2

# Seed
np.random.seed(1)

# Data (assuming that samples are grouped according to their class.)
ng = 100
c = np.array([[[3,2]],[[5,3]],[[5,7]],[[1,4]]])

data=np.repeat(c[0],ng, axis=0)+np.random.normal(0,1,size=(ng,d))
labels = [0]*ng
for i in range(1,len(c)):
  data = np.concatenate((data,np.repeat(c[i],ng, axis=0)+np.random.normal(0,1,size=(ng,d))))
  labels += [i]*ng

# Number of samples
N = len(data)

# The wavefunction
def Psi(x):
  return -np.sum(np.exp(-np.sum(np.square(np.repeat([x],N, axis=0)-data),axis=1)/(2*sigma**2)))

# The potential
def V(x):
  s = np.sum(np.square(np.repeat([x],N, axis=0)-data),axis=1)
  return np.sum(np.multiply(s,np.exp(-s/(2*sigma**2))))/abs(Psi(x))

########## Visualization ###########

# Coordinates
xmin = -2
xmax = 12
Nx = 100
xs = np.linspace(xmin,xmax,Nx)
x,y =np.meshgrid(xs,xs)

# Plot data
fig = plt.figure()
plt.scatter(data[:,0],data[:,1],alpha=0.5,marker='o',label='data')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('x[0]',fontsize=15)
plt.ylabel('x[1]',fontsize=15)
plt.savefig('data.png')

# Test sigma
for sigma in [0.5,0.75,1.0,1.25,1.5]:
  psi = np.zeros((Nx,Nx))
  v = np.zeros((Nx,Nx))
  for i in range(Nx):
    for j in range(Nx):
      psi[i,j] = abs(Psi([xs[i],xs[j]]))
      v[i,j] = V([xs[i],xs[j]])
  psi = psi/np.max(psi)
  v = v/np.max(v)
  # Plot
  # psi
  fig = plt.figure()
  plt.contour(x, y, psi.T,levels=20,linewidth=0.5,alpha=0.6)
  plt.scatter(data[:,0],data[:,1],alpha=0.8,marker='o',c='k',s=7,label='data')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('x[0]',fontsize=15)
  plt.ylabel('x[1]',fontsize=15)
  plt.savefig('s-%.2f-psi.png'%sigma)
  # log psi
  fig = plt.figure()
  plt.contour(x, y, np.log(psi.T),levels=20,linewidth=0.5,alpha=0.6)
  plt.scatter(data[:,0],data[:,1],alpha=0.8,marker='o',c='k',s=7,label='data')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('x[0]',fontsize=15)
  plt.ylabel('x[1]',fontsize=15)
  plt.savefig('s-%.2f-logpsi.png'%sigma)
  # v
  fig = plt.figure()
  plt.contour(x, y, v.T,levels=20,linewidth=0.5,alpha=0.6)
  plt.scatter(data[:,0],data[:,1],alpha=0.8,marker='o',c='k',s=7,label='data')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('x[0]',fontsize=15)
  plt.ylabel('x[1]',fontsize=15)
  plt.savefig('s-%.2f-v.png'%sigma)
  # log v
  fig = plt.figure()
  plt.contour(x, y, np.log(v.T),levels=20,linewidth=0.5,alpha=0.6)
  plt.scatter(data[:,0],data[:,1],alpha=0.8,marker='o',c='k',s=7,label='data')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('x[0]',fontsize=15)
  plt.ylabel('x[1]',fontsize=15)
  plt.savefig('s-%.2f-logv.png'%sigma)

############## classification ################

def classify(func):
  predict = np.zeros(N)
  for i in range(N):
    predict[i] = np.sum(np.around(fmin_bfgs(func,data[i],gtol=0.000001,disp=False),decimals=1))
  return predict

def compare(func):
  # classification
  predict = classify(func)
  # Define classes
  predicted_classes = np.sort(np.unique(predict))
  actual_classes = np.sort(np.unique(labels))
  # Evaluation
  evaluation_table = np.zeros((len(actual_classes),len(predicted_classes)))
  m = 0
  for i in range(len(actual_classes)):
    sub_data = data[np.where(labels==actual_classes[i])]
    unique, counts = np.unique(predict[m:m+len(sub_data)], return_counts=True)
    for u in range(len(unique)):
      j =  np.where(predicted_classes==unique[u])[0]
      evaluation_table[i,j] = counts[u]
    m += len(sub_data)
  return evaluation_table

def score(table):
  x,y=np.shape(table)
  if x>y:
    return np.sum(np.diag(table[np.argmax(table,axis=0),:]))/N*100
  else:
    return np.sum(np.diag(table[:,np.argmax(table,axis=1)]))/N*100

# vary sigma
sigmas = np.linspace(0.1,2,50)
scores_p = np.zeros(len(sigmas))
scores_v = np.zeros(len(sigmas))

for i in range(len(sigmas)):
  sigma = sigmas[i]
  scores_p[i] = score(compare(Psi))
  scores_v[i] = score(compare(V))
  print(i)

# Plot
fig = plt.figure()
plt.plot(sigmas,scores_p,linewidth=3,label='Psi',alpha=0.8)
plt.plot(sigmas,scores_v,'--',linewidth=3,label='V',alpha=0.8)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('sigma',fontsize=15)
plt.ylabel('score (%)',fontsize=15)
plt.savefig('scores.png')


