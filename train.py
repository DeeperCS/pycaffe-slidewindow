# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:40:21 2016

@author: joe
"""
import sys
sys.path.insert(0, '~/github/caffe/python')

import os
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import matplotlib.pyplot as plt

def mirror_edges(X, nPixels):
    assert(nPixels>0)
    
    [s, height, width] = X.shape
    Xm = np.zeros([s, height+2*nPixels, width+2*nPixels], dtype=X.dtype)
    Xm[:, nPixels:height+nPixels, nPixels:width+nPixels] = X
    
    for i in range(s):
        # top left corner
        Xm[i, 0:nPixels, 0:nPixels] = np.fliplr(np.flipud(X[i, 0:nPixels, 0:nPixels]))
        # top right corner
        Xm[i, 0:nPixels, width+nPixels:width+2*nPixels] = np.fliplr(np.flipud(X[i, 0:nPixels, width-nPixels:width]))    
        # bottom left corner
        Xm[i, height+nPixels:height+2*nPixels, 0:nPixels] = np.fliplr(np.flipud(X[i, height-nPixels:height, 0:nPixels]))  
        # bottom right corner        
        Xm[i, height+nPixels:height+2*nPixels, width+nPixels:width+2*nPixels] = np.fliplr(np.flipud(X[i, height-nPixels:height, width-nPixels:width]))
        # top
        Xm[i, 0:nPixels, nPixels:width+nPixels] = np.flipud(X[i, 0:nPixels, 0:width])
        # bottom
        Xm[i, height+nPixels:height+2*nPixels, nPixels:width+nPixels] = np.flipud(X[i, height-nPixels:height, 0:width])
        # left
        Xm[i, nPixels:height+nPixels, 0:nPixels] = np.fliplr(X[i, 0:height, 0:nPixels])
        # right
        Xm[i, nPixels:height+nPixels, width+nPixels:width+2*nPixels] = np.fliplr(X[i, 0:height, width-nPixels:width])
    return Xm
    
    
def pixel_generator(Y, tileRadius, batchSize):
    [s, height, width] = Y.shape
    yAll = np.unique(Y)
    assert(len(yAll)) > 0
    
    # using a bitMask to substract the edges
    bitMask = np.ones(Y.shape, dtype=np.bool)
    bitMask[:, 0:tileRadius, :] = 0
    bitMask[:, height-tileRadius:height,:] = 0
    bitMask[:, :, 0:tileRadius] = 0
    bitMask[:, :, width-tileRadius:width] = 0
    
    # determine how many instances of each class to report
    count = {}
    for y in yAll:
        count[y] = np.sum((Y==y) & bitMask)
    print('[train:]pixels per class label is: {} => {}'.format(count.keys(), count.values()))
    
    sampleCount = np.min(count.values())
    
    Idx = np.zeros([0, 3], dtype=np.int32)
    for y in yAll:
        tup = np.nonzero((Y==y) & bitMask)    # Iterate all slices
        Yi = np.column_stack(tup)    # Idx[1]:height Idx[2]:width
        # minor class will be all sampled
        # major class will be random sampled by the shuffle() below
        np.random.shuffle(Yi)
        Idx = np.vstack((Idx, Yi[:sampleCount,:]))
        
    # one last shuffle to mix all the classes together
    np.random.shuffle(Idx)
    
    #return
    for i in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0]-i)
        yield Idx[i:i+nRet,:]
        
    
if __name__=='__main__':
    trainDisplayInternal = 20
    #######################
    # 1.load TIF to X
    #######################
    X = np.load('trainX.npy')
    Y = np.load('trainY.npy')
    
    #######################
    # 2. mirror to Xm
    #######################
    tileEdge = 65  
    tileRadius = tileEdge//2
    
    Xm = mirror_edges(X, tileRadius)  # (5, 349, 493)
    Ym = mirror_edges(Y, tileRadius)
    
#    plt.figure()
#    plt.imshow(Xm[0,...], cmap='gray')
#    
#    plt.figure()
#    plt.imshow(X[0,...], cmap='gray')
#    
#    plt.figure()
#    plt.imshow(Y[0,...], cmap='gray')
#    
#    plt.figure()
#    plt.imshow(Ym[0,...], cmap='gray')
    
    #######################
    # 3.yield idx
    #######################
    print('yAll is {}'.format(np.unique(Y)))
    
    batchDim = [100, 1, tileEdge, tileEdge]
    print('batch-size:{}'.format(batchDim))
    
    X_batch = np.zeros(batchDim, dtype=np.float32)
    Y_batch = np.zeros(batchDim[0], dtype=np.float32)

    #######################
    # 4.trainng
    #######################
    # Solver parameter
    solverFile = 'solver.prototxt'
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFile).read(), solverParam)
    # net parameter
    netFile = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFile).read(), netParam)
    # model storage
    outDir = solverParam.snapshot_prefix
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    
    #--------------
    # Create Caffe solver
    #--------------
#    solver = caffe.SGDSolver(solverFile)
    solver = caffe.get_solver(solverFile)
    
    for name, blobs in solver.net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print('{}[{}] : {}'.format(name, bIdx, b.data.shape))
    
    # Iteration for a max_iter and record epoch
    currIter = 0
    currEpoch = 0  
    while currIter < solverParam.max_iter:
        currEpoch += 1    # one iteration of the iterator for a epoch
        iterator = pixel_generator(Ym, tileRadius, batchDim[0])
        # iterate for a epoch (All training data for one time)
        for Idx in iterator:
            currIter += 1  # one batch for a iteration
            if currIter > solverParam.max_iter:
                break
            
            Xi = np.zeros(batchDim, dtype=np.float32)
            yi = np.zeros([batchDim[0],], dtype=np.float32)
            # Extract tiles and labels from axis information from Idx(prepare for a batch)
            for j in range(Idx.shape[0]):
                left = Idx[j, 1] - tileRadius
                right = Idx[j, 1] + tileRadius + 1
                top = Idx[j, 2] - tileRadius
                bottom = Idx[j, 2] + tileRadius + 1
                Xi[j, 0, :, :] = Xm[Idx[j, 0], left:right, top:bottom]    # Idx[j, 0]: slice id
                yi[j] = Ym[Idx[j, 0], Idx[j, 1], Idx[j, 2]]
            
            # when a batch data prepared, put it to caffe
            solver.net.set_input_arrays(Xi, yi)
            
            # launch one step for gradient decent
            solver.step(1)
            
            # get loss and accuracy
            loss = float(solver.net.blobs['loss'].data)
            accuracy = float(solver.net.blobs['accuracy'].data)
            
            if currIter%trainDisplayInternal==0:
                print('[train:] currEpoch:{}, iteration:{}, loss:{}, accuracy:{}'.format(currEpoch, currIter, loss, accuracy))

    print('training completed');
