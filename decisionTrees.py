from sklearn.ensemble import RandomForestClassifier
import numpy

def main():
    train = numpy.loadtxt('train.csv', dtype="float" ,delimiter=",")
    train = train.astype(float)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    rf.fit(train, target)

    test = numpy.loadtxt('test.csv', dtype="float" ,delimiter=",")
    test = test.astype(float)

    clases = rf.predict(test)
    indices = numpy.indices(clases.shape)+1
    result = numpy.vstack([numpy.hstack([indices[0][:, numpy.newaxis], clases[:, numpy.newaxis]])])
    
    with open("resultTR.csv", "wb") as f:
        f.write(b'id,Solution\n')
        numpy.savetxt(f, result, fmt='%i', delimiter=",")

if __name__=="__main__":
    main()