#Import libraries
from pylab import *
import scipy.sparse.linalg as linal
import cv2
import random as rn
from graph import *

                                    #Partition the graph
def gpart(D,W,e=2) :
    '''
    I/p to function
    D : matrix D with sum of weights of each node
    W : sparse matrix consisting of the values,indices of weighted graph
    e : No. of Eigen values needed
    O/p of function
    '''
                                    #Solve Eigen system
    eigval,eigvec = linal.eigsh(D-W,e,M=D)
    eigvecSec = eigvec[:,1]           #Second smallest Eigen vector
    thres = median(eigvecSec)
    athres = np.where(eigvecSec >= thres)[0]
    bthres = np.where(eigvecSec < thres)[0]
   
    return athres,bthres
    
def ImgSeg(img) :
    '''
    I/p to function
    img : Image which has to be segmented
    O/p of function
    imgSeg : 2-way segmented image
    '''
    imgSeg = np.zeros(shape(img))
    graph = wgraph(img)
    D = Dgraph(graph)
    athres,bthres = gpart(D,W)
                                    #Obtain the indices
    athresc = athres%shape(imgSeg)[1]
    athresr = (athres-athresc)//3
    bthresc = bthres%shape(imgSeg)[1]
    bthresr = (bthres-bthresc)//3
                                    #Assign intensities to the partitions
    intensity = rn.sample(range(0,255),2)
    imgSeg[athresr,athresc] = intensity[0]
    imgSeg[bthresr,bthresc] = intensity[1]
    
    return imgSeg

imgSeg = np.zeros((3,3))
athres,bthres = gpart(D,graph)
print(athres,bthres)
athresc = athres%shape(imgSeg)[1]
athresr = (athres-athresc)//3
bthresc = bthres%shape(imgSeg)[1]
bthresr = (bthres-bthresc)//3
print(athresr,athresc)
                                #Assign intensities to the partitions
intensity = rn.sample(range(0,255),2)
imgSeg[athresr,athresc] = intensity[0]
imgSeg[bthresr,bthresc] = intensity[1]   
print(imgSeg) 
print(src1[:3,:3])
#print(gpart(diag([1.0,2.0,3.0,4.0]),np.arange(16).reshape(4,4).T.astype(float)))
