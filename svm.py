import numpy
from sklearn import svm, preprocessing, decomposition, grid_search, cross_validation


def main():
	
	train = numpy.loadtxt('train.csv', dtype="float" ,delimiter=",")
	train = train.astype(float)

	target = numpy.loadtxt('trainLabels.csv', dtype="int" ,delimiter=",")
	
	pca = decomposition.PCA(n_components=12)
	train = pca.fit_transform(train)
	scaler = preprocessing.StandardScaler().fit(train)

#	clf = svm.SVC(kernel='rbf', gamma=0.25, coef0=1.1, cache_size=8000)
#	clf.fit(scaler.transform(train), target)
	tuned_parameters = [{'kernel' : ['rbf'], 'gamma': [1, 0.3, 0.27, 0.26, 0.25, 0.24, 0.23, 0.2, 0.15, 0.1, 0.05, 1e-2, 1e-3, 1e-4], 'C': [0.5, 0.9, 1, 1.1, 1.2, 2]}, 
                  {'kernel' : ['poly'], 'degree' : [2, 3, 4, 5, 7, 8, 9], 'C' : [0.001, 0.01, 0.1, 1, 10, 100]}]

	cvk = cross_validation.StratifiedKFold(target, n_folds=5)              
	clf = grid_search.GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=4 , cv=cvk, verbose=2).fit(scaler.transform(train), target)

	test = numpy.loadtxt('test.csv', dtype="float" ,delimiter=",")
	test = test.astype(float)
	test = pca.transform(test)

	clases = clf.predict(scaler.transform(test))

	indices = numpy.indices(clases.shape)+1
	result = numpy.vstack([numpy.hstack([indices[0][:, numpy.newaxis], clases[:, numpy.newaxis]])])


	with open("resultSVM.csv", "wb") as f:
		f.write(b'id,Solution\n')
		numpy.savetxt(f, result, fmt='%i', delimiter=",")

	print clf.best_estimator_	


if __name__ == '__main__':
	main()