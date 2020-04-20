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
    y = np.zeros(shape(eigvecSec))  #Divide into partitions based on thres
    athres = np.where(eigvecSec >= thres)[0]
    bthres = np.where(eigvecSec < thres)[0] #Define k as 'b' in paper
    k = (sum(D.tocsr()[athres,athres]))/(sum(D.tocsr()[bthres,bthres]))
    y[athres] = 1
    y[bthres] = -k                  #Use formula to obtain Ncut
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
    eigvecSec = eigvec[:,1]         #Second smallest Eigen vector
    thres = scipy.optimize.fmin(func = Ncut,x0 = median(eigvecSec),args = (eigvecSec,D,W))[0]
    ncut = Ncut(thres,eigvecSec,D,W)#Obtain Ncut value and divide based on thres
    athres = np.where(eigvecSec >= thres)[0]
    bthres = np.where(eigvecSec < thres)[0]
    #print(len(athres),len(bthres)) #Used to calculate Arthres
   
    return athres,bthres,ncut

                                #Recursively obtain all the partitions
def recpart(part,W,start,Arthres=1500,NCthres=0.03) :
    '''
    I/p to function
    W : Weighted graph used to extract submatrices
    partition : final set of partitions
    newpart : indices of the higher partition which has to be broken further
    Athres : min area threshold
    NCthres : min Ncut threshold
    O/p of function
    partition : final set of partitions
    '''
    print("Entered recpart")
                                #Reduce reduntant comuptation at start
    if start == True :
        W0 = W
        start = False
    else :                      #Take the required portion of weighted graph
                                #Obtain row and column indices
        row = part.repeat(len(part))
        col = array(list(part)*len(part))
                                #Convert to lil so that you can address required indices
        W0 = W.tolil()[row,col].reshape(len(part),len(part))
        W0 = W0.tocsr()    
    D0 = Dgraph(W0)             #Divide part into 2 partitions
    athres,bthres,ncut = gpart(D0,W0)
    partition = []
    if ncut <= NCthres and ncut >= 0.0001:      #If there is sufficient dissimilarity
        status = "NA"           #Satisfies Ncut check for area condition
        if len(athres) >= Arthres and len(bthres) >= Arthres :  
                                #Satisfies area and Ncut constraint                     
            apartition,astat = recpart(athres,W,start)
            bpartition,bstat = recpart(bthres,W,start)
        else :                  #Area not satisfied => Don't divide
            status = "A"
        
        if status == "A" :   
            return partition,status
            
        if astat == "A" :       #If athres cannot be divided further
            partition.append(part[athres])
        elif astat == "NA" :    #If athres can be divided further
            partition = partition + apartition
        if bstat == "A" :       #If bthres cannot be divided further
            partition.append(part[bthres])
        elif bstat == "NA" :    #If bthres can be divided further
            partition = partition + bpartition
    else :                      #Does not satisfy Ncut     
        status = "A"
        
    return partition,status
    
def ImgSeg(img) :
    '''
    I/p to function
    img : Image which has to be segmented
    O/p of function
    imgSeg : 2-way segmented image
    '''
    print("Entered ImgSeg")
    
    graph = wgraph(img)             #Generate W
    D = Dgraph(graph)               #Generate D
                                    #Generate partition
    partition,status = recpart(arange(len(img.flatten())),graph,True)
                                    #Output shape to hold all partitions concatanated
    imgSeg = np.zeros((shape(img)[0],int((len(partition)+1)*shape(img)[1])))                    
    imgSeg[:,:shape(img)[1]] = img  #First section is the original image
    for i in range(0,len(partition)) :
        coloffset = shape(img)[1]*(i+1)
        athres = partition[i]
                                    #Generate row col indices from W indices
        athresc = athres%shape(img)[1]
        athresr = (athres-athresc)//shape(img)[1]
                                    #Assign each partition to output
        imgSeg[athresr,coloffset+athresc] = img[athresr,athresc]
                                    #Plot the output
    plt.imshow(imgSeg,cmap = "gray")
    plt.axis("off")
    plt.show()
    
    return imgSeg
                                    #Executing code
src = src1
#src = src2
plt.imshow(src,cmap = "gray")       #Plot input
plt.axis("off")
plt.show()        
op = ImgSeg(src)                    #Run the entire code
