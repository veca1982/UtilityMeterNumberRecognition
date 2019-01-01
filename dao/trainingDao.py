__author__ = 'Krtalici'
import pickle as pkl
import numpy as np

def writeTrainingData( imagesInVectorForm ) :
    np.savetxt('test.txt', imagesInVectorForm)

def writeTargets( ) :
    outcomes = open('trainingData/target.pkl', 'w')

    #target = np.array([0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0])
    target = np.array([0,1,0,0,1])

    pkl.dump(target, outcomes)
    outcomes.close()

def readTrainingData() :
    imagesInVectorForm = np.loadtxt('test.txt')
    imagesInVectorForm = np.delete(imagesInVectorForm, 0, 0)
    print "Size [%d,%d]"%imagesInVectorForm.shape

    #matrica trening slika, svaka slika pretvorena pri zapisivanju u datoteku u vektor
    return imagesInVectorForm

def readTrainingTarget() :
    pkl_file = open('trainingData/target.pkl', 'r')
    target = pkl.load(pkl_file)
    pkl_file.close()
    #matrica trening slika, svaka slika pretvorena pri zapisivanju u datoteku u vektor
    return target

def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass
