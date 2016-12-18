import numpy as np

def main():
	a = np.concatenate((np.eye(5), np.fliplr(np.eye(5))))
	a = np.concatenate((a,a))
	m = np.concatenate((np.ones(10), np.zeros(10)))
	print "Matrix before mask", a
	a = a[m>0,:]
	print "Mask", m
	print "Matrix after mask", a
	A = np.dot(a[:-1].T,a[1:])
	print "2-gram counts", A

	A_sum = np.sum(A, axis=1)
	print "2-gram counts sum:", A_sum

	A_norm = A / A_sum.T 
	print "2-gram counts normalized:", A_norm

if __name__ == '__main__':
	main()