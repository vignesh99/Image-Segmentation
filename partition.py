#Import libraries
from pylab import *
import scipy.sparse.linalg as linal
import cv2
import random as rn
from graph import *
import scipy.optimize
                                    #Obtain Ncut
def Ncut(thres,eigvecSec,D,W) :
    '''
    I/p to function
    thres : threshold value to divide the regions
    eigvecSec : 2nd eigenvector used to partition the graph
    D : matrix D with sum of weights of each node
    W : sparse matrix consisting of the values,indices of weighted graph
    O/p of function
    Ncut : Normalized cut value
    '''
    y = np.zeros(shape(eigvecSec))
    athres = np.where(eigvecSec >= thres)[0]
    bthres = np.where(eigvecSec < thres)[0]
    #print(D.tocsr()[[1,2],[1,2]])
    k = (sum(D.tocsr()[athres,athres]))/(sum(D.tocsr()[bthres,bthres]))
    #print(k)
    #k = 1
    y[athres] = 1
    y[bthres] = -k
    ncut = (y.T @ (D-W) @ y)/(y.T @ D @ y)
    
    return ncut
                                    #Partition the graph
def gpart(D,W,e=2) :
    '''
    I/p to function
    D : matrix D with sum of weights of each node
    W : sparse matrix consisting of the values,indices of weighted graph
    e : No. of Eigen values needed
    O/p of function
    '''
    print("Entered gpart")
                                    #Solve Eigen system
    eigval,eigvec = linal.eigsh(D-W,e,M=D,which = 'SA')
    eigvecSec = eigvec[:,1]           #Second smallest Eigen vector
    #print(eigvecSec)
    #print(eigval)
    #print(eigvecSec[rn.sample(range(len(eigvecSec)),3)])
    #print("Enter opt")
    thres = scipy.optimize.fmin(func = Ncut,x0 = median(eigvecSec),args = (eigvecSec,D,W))[0]
    print(thres,mean(eigvecSec))
    #thres = median(eigvecSec)
    #thres = mean(eigvecSec)
    #thres = 0
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
    print("Entered ImgSeg")
    imgSeg = np.zeros(shape(img))
    graph = wgraph(img)
    D = Dgraph(graph)
    athres,bthres = gpart(D,graph)
                                    #Obtain the indices
    athresc = athres%shape(imgSeg)[1]
    athresr = (athres-athresc)//shape(imgSeg)[1]
    bthresc = bthres%shape(imgSeg)[1]
    bthresr = (bthres-bthresc)//shape(imgSeg)[1]
                                    #Assign intensities to the partitions
    #intensity = rn.sample(range(0,255),2)
    intensity = [mean(img[athresr,athresc]),mean(img[bthresr,bthresc])]
    #intensity = [min(img[athresr,athresc]),max(img[bthresr,bthresc])]
    imgSeg[athresr,athresc] = intensity[0]
    imgSeg[bthresr,bthresc] = intensity[1]
    #imgSeg[athresr,athresc] = img[athresr,athresc]
    #imgSeg[bthresr,bthresc] = img[bthresr,bthresc]
    
    plt.imshow(imgSeg,cmap = "gray")
    plt.show()
    
    return imgSeg

#src = src1[:200,:200]
#src = src1[150:,100:]
#src = src1
src = src2
#src = np.zeros((100,100))
#src[50:,40:90] = 255
#src[50:75,:25] = 100
#src[:40,30:] = 200
#src = src[:,50:]
plt.imshow(src,cmap = "gray",vmin=0, vmax=1)
plt.show()        
op = ImgSeg(src)
#r = 0
#img = src1[r:r+3,r:r+3]
#imgSeg = np.zeros((3,3))
#print('{0:.25f}'.format(graph[4,5]))
#print(D-graph.todense())
#athres,bthres = gpart(D,graph)
#print(athres,bthres)
#athresc = athres%shape(imgSeg)[1]
#athresr = (athres-athresc)//3
#bthresc = bthres%shape(imgSeg)[1]
#bthresr = (bthres-bthresc)//3
#print(athresr,athresc)
                                #Assign intensities to the partitions
#intensity = [mean(img[athresr,athresc]),mean(img[bthresr,bthresc])]
#intensity = [min(img[athresr,athresc]),max(img[bthresr,bthresc])]
#imgSeg[athresr,athresc] = intensity[0]
#imgSeg[bthresr,bthresc] = intensity[1] 
#plt.imshow(imgSeg)
#plt.show()  
#print(imgSeg) 
#print(img)
#print(gpart(diag([1.0,2.0,3.0,4.0]),np.arange(16).reshape(4,4).T.astype(float)))
