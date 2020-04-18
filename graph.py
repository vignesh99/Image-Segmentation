#Import libraries
from pylab import *
from scipy.sparse import *
import cv2
from numpy.lib import stride_tricks
from skimage.transform import resize

#Load image for test
path1  = r"img1.jpg"
src1 = cv2.imread(path1,0)
src1 = resize(src1, (src1.shape[0] // 2, src1.shape[1] // 2),anti_aliasing=True)
#path2  = r"thres_palmleaf1.png"
path2  = r"img6.jpg"
src2 = cv2.imread(path2,0)
src2 = resize(src2, (src2.shape[0] // 4, src2.shape[1] // 4),anti_aliasing=True)

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
    print("Entered indRad")
    V = len(img.flatten())
    Nrow = shape(img)[0]
    Ncol = shape(img)[1]
    rowvec = (np.arange(Nrow).repeat(Ncol)).reshape(1,V)
    colvec = array(list(np.arange(Ncol))*Nrow).reshape(1,V)
    one = np.ones((1,int(2*r+1),int(2*r+1)))
    #print(shape(rowvec))
    #print((one),shape(array([rowvec,colvec]).T))
    blkcenr = (one.T @ rowvec).T
    blkcenc = (one.T @ colvec).T
    #print((blkcenr,blkcenc)) 
                                #Find the indices in radius r
    ind = -(r+2)*np.ones((2,shape(img)[0]+int(2*r),shape(img)[1]+int(2*r)))
    ind[:,r:-r,r:-r] = np.indices(shape(img))
    indr = ind[0]
    indc = ind[1]
    strides = indr.strides*2
    size = int(2*r)+1
    dim = (indr.shape[0] - size + 1, indr.shape[1] - size + 1, size, size)
    patchr = stride_tricks.as_strided(indr, shape=dim, strides=strides)
    patchr = patchr.reshape(V,size,size)
    
    strides = indc.strides*2
    size = int(2*r)+1
    dim = (indc.shape[0] - size + 1, indc.shape[1] - size + 1, size, size)
    patchc = stride_tricks.as_strided(indc, shape=dim, strides=strides)
    patchc = patchc.reshape(V,size,size)
    
    diffr = patchr - blkcenr
    diffc = patchc - blkcenc
    
    indnorm = norm(array([diffr,diffc]),axis = 0)   #Incdices with norm within r
    fullind = np.where((indnorm <= r))
    indcen = fullind[0]                         #The centre indices values with neighbours within their radius
    indcenc = indcen%shape(img)[1]              #Obtain indcenc
    indcenr = (indcen-indcenc)//shape(img)[1]   #Obtain indcenr
    
    indrow = fullind[1]+indcenr-r               #Obtain indrow
    indcol = fullind[2]+indcenc-r               #Obtain indcol
    indices = indrow*shape(img)[1] + indcol     #Suitable format for sparse matrix generation
        
    indptr = cumsum(np.unique(fullind[0],return_counts= True)[1])
    indptr = array([0] + list(indptr))          #Obtain indptr
    
    #print(shape(img)[1])
    #print(np.where(indrow >= shape(img)[0]))
    #print(np.where(indcenr >= shape(img)[0]))
    #print(np.where(indcol >= shape(img)[1]))
    #print(np.where(indcenc >= shape(img)[1]))
    #print(indrow)
    #print(indcol)
    #print(indices)
    #print(indcen)
    #print(indcenr)
    #print(indcenc)
    #print(indptr)
    #print((fullind)[0])
    #print((indnorm[0]))
    #print(patchr[3],"\n",patchc[3])
    #index = np.zeros((V,5,5))
    #print(shape(index))
    #print(np.where(norm(ind-array([[ind[:,5,6][0]*one],[ind[:,5,6][1]*one]]),axis=0)<5))
    
    return indptr,indrow,indcol,indcenr,indcenc,indices
   
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
    print("Entered wval")
                                #Image intensity contribution to weight
    imgexp = np.exp(-1*((img[indrow,indcol] - img[indcenr,indcenc])**2)/(sigI**2))
    #print(-1*((img[indrow,indcol] - img[indcenr,indcenc])**2))
    #print(imgexp)
                                #Index distance contribution to weight
    indexp = np.exp(-(norm(array([indrow,indcol])-array([indcenr,indcenc]),axis=0)**2)/(sigX**2))
    wgt = imgexp*indexp         #Define the weight values
    
    return wgt                                   
    
                                #Generate weighted graph
def wgraph(img,sigI=0.1,sigX=4,r=5) :
    '''
    I/p to function
    img : Image whose information is to be extracted
    sigI : Image variance
    sigX : Index variance
    r : measure of neighbourhood
    O/p of function
    graph : sparse matrix consisting of the values,indices of weighted graph
    '''
    print("Entered wgraph")
    V = len(img.flatten())
    indptr,indrow,indcol,indcenr,indcenc,indices = indRad(img,r)
    wgt = wval(img,indrow,indcol,indcenr,indcenc,sigI,sigX)
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
    print("Entered Dgraph")
    diagD = np.asarray(csr_matrix.sum(graph,axis = 1)).reshape(-1)
    #print(diagD[0])
    V = len(diagD)
    #print(shape(diagD),type(diagD))
    D = diags(diagD,shape = (V,V))
    
    return D

#Test inputs
#r = 0
#img = src1[r:r+3,r:r+3]
#img = array([[0,0,0],[0,255,0],[200,0,200]])
#indRad(img,1)
#indptr = array([0,3,7,10,14,19,23,26,30,33])
#print(indptr)
#indrow = array([0,0,1,0,0,1,0,0,0,1,0,1,1,2,0,1,1,1,2,0,1,1,2,1,2,2,2,2,1,2,2,1,2])
#indcol = array([0,1,0,0,1,1,2,1,2,2,0,0,1,0,1,0,1,2,1,2,1,2,2,0,0,1,0,1,1,2,1,2,2])
#indices = indrow*3 + indcol
#print(indrow)
#print(indcol)
#print(indices)
#print(sort(indices))
#indcenr = array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2])
#indcenc = array([0,0,0,1,1,1,1,2,2,2,0,0,0,0,1,1,1,1,1,2,2,2,2,0,0,0,1,1,1,1,2,2,2])
#indcen = indcenr*3 + indcenc
#print(indcen)
#print(indcenr)
#print(indcenc) 
#print(len(indrow))
#sigI = 20
#sigX = 5  
#indptr,indrow,indcol,indcenr,indcenc,indices = indRad(img,1)
#wgt = wval(img,indrow,indcol,indcenr,indcenc,sigI,sigX)
#graph1 = csr_matrix((wgt,indices,indptr),shape=(9,9))

#indptr,indrow,indcol,indcenr,indcenc,indices = indRad(img,1)
#wgt = wval(img,indrow,indcol,indcenr,indcenc,sigI,sigX)
#graph = csr_matrix((wgt,indices,indptr),shape=(9,9))
#print(np.where(graph1.todense() != graph2.todense()))
#print(graph1[np.where(graph1.todense() != graph2.todense())])
#print(graph2[np.where(graph1.todense() != graph2.todense())])
#graph = graph1
#D = Dgraph(graph)
#Dsq = csr_matrix.sqrt(D)
#print(1/D)
#print(graph.todense())
#print(D.todense())

