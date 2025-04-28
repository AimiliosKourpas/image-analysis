import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, Caltech101
import matplotlib.pyplot as plt
import numpy as np
import random
import gdown
import hypernetx as hnx
import tarfile
import os
import timm
import math

# Function Definitions
def similarity_lists(features):  #original similarity function
  '''
  Computes the euclidean distance between every pair of features.
  features = list of features extracted from each image
  '''
  T = [[(0,0) for i in range(len(features))] for j in range(len(features))]
  for i in range(len(features)):
    for j in range(i,len(features)):
      
      score = 1/(np.linalg.norm(features[i].cpu()-features[j].cpu())+0.00001) 
      #use the fact that euclidean distance is symmetrical to speed up computation (a-b)^2 = (b-a)^2
      T[i][j]=(score,j) 
      T[j][i]=(score,i) 
  return T

def rank_normalization(features,T): # rank normalization
  '''
  Performs reciprocal rank normalization as defined in the paper 
  ρ_n(i,j) = 2L - (τ_i(j)+τ_j(i)),
  for each list τ in T, then sorts τ.

  features = list of features extracted from each image
  T = list of lists of distances
  '''
  L = len(features)
  for i in range(len(T)): #compute score for each pair
    for j in range(i,len(T)):
      score = 2*L - (T[i][j][0] + T[j][i][0])
      T[i][j] = (score,j)
      T[j][i] = (score,i)
  T = [sorted(t,key = lambda x: x[0]) for t in T]  #sort each sublist
  return T

def make_hyperedges(T,k): #make e_i
  '''
  Creates hyperedges as lists of nodes.
  Hyperedge e_i contains the first k nodes of T[i].

  T = list of sorted similarities for each pair of image features
  k = size of neighborhood
  '''
  E = []  
  for t in T: 
    temp = []
    for p in t[:k]: 
      temp.append(p[1]) 
    E.append(temp)
  return E 

import math

def association(E,V,T,k):
  '''
  Creates the association/incidence matrix r(e_i,v_j)=h(e_i,v_j).
  Note: In our variation, since each hyperedge is centered around a node/image, it is clear that |E| = |V|. 
  Thus, the second parameter V can also be the list of hyperedges, since we only use its length. 

  E = list of hyperedges (each hyperedge is a list of nodes)
  V = list of nodes in the hypergraph
  T = list of sorted similarities for each pair of image features
  k = neighborhood size
  '''
  R = np.zeros((len(E),len(E)))
  for i,e in enumerate(E):  #for each edge
    for v in range(len(V)): #for each vertex
      if v in e:  #if vertex is in the hyperedge
        pos = e.index(v)+1  #get the position (+1 because counting in the paper starts from 1)
        R[i][v] = 1-math.log(pos,k+1)  #compute the weight
      else: #if vertex is not in the hyperedge
        R[i][v] = 0 #weight is 0
  return R


def edge_weights(E,assoc):
  '''
  Computes edge weights as defined in page 6 of the paper.

  E = list of hyperedges
  assoc = association/incidence matrix H
  '''
  w = []
  for i,e in enumerate(E):  #for each hyperedge
    s=0
    for j in e: #for each node in the hyperedge
      s+= assoc[i][j]
    w.append(s)
  return w

def hyperedge_similarities(assoc):  #Hyperedge Similarities
  '''
  Computes pairwise similarity matrix S as defined in page 6 of the paper.
  '''
  H = np.array(assoc)
  Sh = H @ H.T  # @ = matrix multiplication
  Su = H.T @ H
  S = np.multiply(Sh,Su)  # Hadamard product
  return S


def cartesian_product(eq,ei):
  '''
  Creates the cartesian product of 2 hyperedges (lists of nodes)

  eq, ei = hyperedges
  '''
  return np.transpose([np.tile(eq, len(ei)), np.repeat(eq, len(ei))])

def pairwise_similarity_relationship(w,assoc,E):
  '''
  Computes pairwise similarity relationship / membership degrees as defined in page 6 of the paper.

  w = hyperedge weights
  assoc = incidence/association matrix
  E = list of hyperedges
  '''

  # v_i, v_j in e_q^2 (cartesian product)
  #p(e_q,v_i,v_j) = |E| x |e_q^2| 
  
  p = [{} for _ in range(len(E))] #for each hyperedge create a dictionary with node pairs as keys
  for i,e in enumerate(E):
    cp3 = cartesian_product(e,e)
    for (v1,v2) in cp3:
      p[i][(v1,v2)] = w[i]*assoc[i][v1]*assoc[i][v2]
  return p


def make_C(E,p):
  '''
  Computes the similarity based on the cartesian product (page 6).

  E = list of hyperedges
  p = list of membership degrees for each pair in each hyperedge
  '''
  C = np.zeros((len(E),len(E)))
  for i,e in enumerate(E):  #for each hyperedge
    for (v1,v2) in p[i]:  #for each pair in the dict
      C[v1][v2]+=p[i][(v1,v2)]  #compute value
  return C


def affinity_matrix(C,S):
  '''
  Computes final affinity matrix (page 6)
  '''
  return np.multiply(C,S)


def LHRR(features,init_lists,k=3,num_iters=10):
  '''
  Entire LHRR algorithm put together

  features = list of features extracted from each image
  init_lists = initial similarity lists based on euclidean distance
  k = neighborhood size / hyperedge size
  num_iters = for how many iterations the algorithm will run
  '''

  for i in range(num_iters):

    T = rank_normalization(features,init_lists)  #perform rank normalization

    E = make_hyperedges(T,k)  #make hyperedges

    #HG = make_hypergraph(E)  #make hypergraph

    assoc = association(E,E,T,k)  #make association/incidence matrix 
    #There is an explanation above as to why E is passed two times

    w = edge_weights(E,assoc) #compute edge weights

    S = hyperedge_similarities(assoc) #compute hyper-edge similarities

    p = pairwise_similarity_relationship(w,assoc,E) #compute pairwise relationships

    C = make_C(E,p) #make the cartesian product based similarity matrix

    aff = affinity_matrix(C,S)  #compute the final matrix W

    # reshape final matrix so it becomes input of the next iteration
    aff = aff.tolist()

    for i,row in enumerate(aff):
      for j,v in enumerate(row):
        aff[i][j] = (aff[i][j],j) #add information about the index

    T = aff
  return aff


# Libraries and Data Preparation

print("Libraries and all the above are ready!")

url = 'https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp' # URL to the dataset
output = '101_ObjectCategories.tar.gz' # Output file
 
gdown.download(url, output, quiet=False) # Download the dataset

file_path = '/Users/aa/image-analysis/101_ObjectCategories.tar.gz' # Path to the downloaded file
extract_path = '/Users/aa/image-analysis/' # Path to extract the file to

with tarfile.open(file_path, 'r:gz') as tar: # Extract the file
    tar.extractall(extract_path) # Extract to the path
 
transform = transforms.Compose([ # Transform the images
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root='/Users/aa/image-analysis/101_ObjectCategories', transform=transform) # Load the dataset
len(dataset)  # Length of our dataset

dataset = list(dataset)  # Turn to list
random.shuffle(dataset)  # Shuffle the list
keep_number = 3500  # Number of records to keep

dataset = dataset[:keep_number]  # Keep only a certain number of records
len(dataset)

images, labels = map(list, zip(*dataset)) # Separate images and labels

model = timm.create_model('vit_base_patch16_224', pretrained=True) # Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Set the device
model.to(device) # Move the model to the device

for param in model.parameters(): # Freeze the model
    param.requires_grad = False # No need to compute gradients

print(model) # Print the model

model.head = nn.Sequential() # Remove the head

# Feature Extraction

features = [] # List to store the features

for img in images: # Extract features 
    features.append(model(img.unsqueeze(0).to(device)))     # Append the features is taking a while 20 minutes

print(len(features)) # Length of the features

init_lists = similarity_lists(features) # Initial similarity lists

final_ranking = LHRR(features, init_lists, k=3, num_iters=3)  # Final matrix W 

# Evaluation

query_index = 1 #second image
retrieved = []  #keep indexes and scores of relevant images here
for (score,i) in final_ranking[query_index]:  #search first row of W
  if score!=0:  #if score is non zero
    retrieved.append((score,i))
retrieved = sorted(retrieved,key = lambda x: x[0],reverse=True) #sort by score

print(retrieved)

def precision(query_index, final_ranking, labels, k=5):
    '''
    Compute precision using labels.

    query_index: Index of the query image.
    final_ranking: Affinity matrix W with final results.
    labels: Image labels.
    k: Number of retrieved images to keep.
    '''

    retrieved = [(score, i) for score, i in final_ranking[query_index] if score != 0]
    retrieved = sorted(retrieved, key=lambda x: x[0], reverse=True)[1:k + 1]

    true_label = labels[query_index]
    correct_count = sum(1 for _, ix in retrieved if labels[ix] == true_label)

    return correct_count / len(retrieved)

def recall(query_index, final_ranking, labels):
    '''
    Compute recall using labels. 
    
    query_index: Index of the query image. 
    final_ranking: Affinity matrix W with final results.
    labels: Image labels.
    '''
    ixs = [i for score, i in final_ranking[query_index] if score != 0]
    ixs.sort(reverse=True)

    true_label = labels[query_index]
    correct_count = sum(1 for ix in ixs[1:] if labels[ix] == true_label)

    db_count = labels.count(true_label)
    
    return correct_count / (db_count - 1) if db_count > 1 else 0

# Results

q_img = 8 # Query image by index (0-3499) 
p = precision(q_img, final_ranking, labels)
print("Precision for image " + str(q_img) + " is:", p)

q_img = 8 # Query image by index (0-3499)
r = recall(q_img, final_ranking, labels)
print("Recall for image " + str(q_img) + " is:", r)



