import numpy

def main():
	array = numpy.loadtxt('resultSVM.csv', dtype="int" ,delimiter=",",skiprows=1)
	array = array[:,1]
	array2 = numpy.loadtxt('resultSVM_grid.csv', dtype="int" ,delimiter=",",skiprows=1)
	array2 = array2[:,1]
	counter = 0;
	for i in range(len(array)):
		if array[i] != array2[i]:
			counter += 1

	print counter


if __name__ == '__main__':
	main()