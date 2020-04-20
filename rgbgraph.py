#Import libraries
from pylab import *
from scipy.sparse import *
import cv2
from numpy.lib import stride_tricks
from skimage.transform import resize

#Load image for test
path1  = r"rgb1.jpg"
src1 = cv2.imread(path1)
src1 = resize(src1, (src1.shape[0] // 4, src1.shape[1] // 4),anti_aliasing=True)[:,:,::-1]
path2  = r"rgb9.jpg"
src2 = cv2.imread(path2)
src2 = resize(src2, (src2.shape[0] // 4, src2.shape[1] // 4),anti_aliasing=True)[:,:,::-1]

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
    V = int(shape(img)[0]*shape(img)[1])
    Nrow = shape(img)[0]
    Ncol = shape(img)[1]
    rowvec = (np.arange(Nrow).repeat(Ncol)).reshape(1,V)
    colvec = array(list(np.arange(Ncol))*Nrow).reshape(1,V)
    one = np.ones((1,int(2*r+1),int(2*r+1)))
    blkcenr = (one.T @ rowvec).T
    blkcenc = (one.T @ colvec).T
                                #Find the indices in radius r
    ind = -(r+2)*np.ones((2,shape(img)[0]+int(2*r),shape(img)[1]+int(2*r)))
    ind[:,r:-r,r:-r] = np.indices(shape(img[:,:,0]))
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
    imgexp = np.exp(-1*(norm(img[indrow,indcol] - img[indcenr,indcenc],axis=1)**2)/(sigI**2))
                                #Index distance contribution to weight
    indexp = np.exp(-(norm(array([indrow,indcol])-array([indcenr,indcenc]),axis=0)**2)/(sigX**2))
    wgt = imgexp*indexp         #Define the weight values
    
    return wgt                                   
    
                                #Generate weighted graph
def wgraph(img,sigI=0.1,sigX=4,r=7) :
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
    V = int(shape(img)[0]*shape(img)[1])
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
    V = len(diagD)
    D = diags(diagD,shape = (V,V))
    
    return D
