__author__ = 'Krtalici'
from sklearn import svm


class Clasiffier:
    clf = svm.SVC()

    def trainClasifier( self, X, y ) :
        Clasiffier.clf.fit(X, y)

    def predict( self, imageInVectorForm ):
        return Clasiffier.clf.predict( imageInVectorForm )
