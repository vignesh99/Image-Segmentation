#Import libraries
from pylab import *
from scipy.sparse import *
import cv2

#Load image for test
path1  = r"img1.jpg"
src1 = cv2.imread(path1,0)
path2  = r"img2.jpg"
src2 = cv2.imread(path2,0)

                                #Get indices of radius R
def indRad(img,r) :
    '''
    I/p to function
    img : Image whose information is to be extracted
    r : measure of neighbourhood
    O/ps of function
    indptr : Gives which set of indices is for which set of rows
    indrow : 1D array containing row values needed to be filled by wgt matrix
    indcol : 1D array containing column values needed to be filled by wgt matrix
    indcenr : 1D array containing the centre pixel row indices around which weight is found
    indcenc : 1D array containing the centre pixel column indices around which weight is found
    indices : Suitable format for sparse matrix generation
    '''
    V = len(img.flatten())
                                #Find the indices in radius r
    ind = np.indices(shape(img))
    one = np.ones(shape(img))
    index = np.zeros((V,5,5))
    print(shape(index))
    #print(np.where(norm(ind-array([[ind[:,5,6][0]*one],[ind[:,5,6][1]*one]]),axis=0)<5))
    
    indices = indrow*3 + indcol #Suitable format for sparse matrix generation
   
                                #Find the weight function    
def wval(img,indrow,indcol,indcenr,indcenc,sigI,sigX) :
    '''
    I/ps to function
    img : Image whose information is to be extracted
    indrow : 1D array containing row values needed to be filled by wgt matrix
    indcol : 1D array containing column values needed to be filled by wgt matrix
    indcenr : 1D array containing the centre pixel row indices around which weight is found
    indcenc : 1D array containing the centre pixel column indices around which weight is found
    sigI : Image variance
    sigX : Index variance
    O/p of function
    wgt : weight values of the graph corresponding to the indices
    '''
                                #Image intensity contribution to weight
    imgexp = np.exp(-1*((img[indrow,indcol] - img[indcenr,indcenc])**2)/(sigI**2))
    #print(-1*((img[indrow,indcol] - img[indcenr,indcenc])**2))
    #print(imgexp)
                                #Index distance contribution to weight
    indexp = np.exp(-(norm(array([indrow,indcol])-array([indcenr,indcenc]),axis=0)**2)/(sigX**2))
    wgt = imgexp*indexp         #Define the weight values
    
    return wgt                                   
    
                                #Generate weighted graph
def wgraph(img,sigI=9,sigX=4,r=5) :
    '''
    I/p to function
    img : Image whose information is to be extracted
    sigI : Image variance
    sigX : Index variance
    r : measure of neighbourhood
    O/p of function
    graph : sparse matrix consisting of the values,indices of weighted graph
    '''
    V = len(img.flatten())
    indptr,indrow,indcol,indcenr,indcenc,indices = indRad(img,r)
    wgt = wval(indrow,indcol,indcenr,indcenc,sigI,sigX)
                                    #Define the weight-sparse matrix
    graph = csr_matrix((wgt,indices,indptr),shape=(V,V))
        
    return graph
                                    #Generate D matrix
def Dgraph(graph) :
    '''
    I/p to function
    graph : Sparse matrix representing the weighted graph
    O/p of function
    D : matrix D with sum of weights of each node
    '''
    diagD = np.asarray(csr_matrix.sum(graph,axis = 1)).reshape(-1)
    #print(diagD[0])
    V = len(diagD)
    #print(shape(diagD),type(diagD))
    D = diags(diagD,shape = (V,V))
    
    return D

#Test inputs
indptr = array([0,2,5,7,10,14,17,19,22,24])
indrow = array([0,1,0,1,0,0,1,0,1,2,0,1,1,2,0,1,2,1,2,2,1,2,2,1])
indcol = array([1,0,0,1,2,1,2,0,1,0,1,0,2,1,2,1,2,0,1,0,1,2,1,2])
indices = indrow*3 + indcol
indcenr = array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2])
indcenc = array([0,0,1,1,1,2,2,0,0,0,1,1,1,1,2,2,2,0,0,1,1,1,2,2])
#print(len(indrow))
sigI = 9
sigX = 4    
wgt = wval(src1[:3,:3],indrow,indcol,indcenr,indcenc,sigI,sigX)
graph = csr_matrix((wgt,indices,indptr),shape=(9,9))
D = Dgraph(graph)
#print(graph.todense())
#print(D)

