import numpy as np
#print help(np.floor)
#print dir(np.zeros((1,1)))
#exit()

VERBOSE = True

#removes correaltions
class DistrInterface(object):
    """ Measures and mimics correlation ininput data"""
    def energy(self, state):
        pass

    def sample(self):
        #sample freely, no input condisioned (clamped)
        pass

    def sample_input(self, x):
        #sample given input
        pass

    def marginal_input(self, x):
        #marginal given input
        pass

    def get_dual(self):
        return self.dual
#
    def sample_correlation(self):
        """ Used for training"""
        pass

class Boltzmann1(DistrInterface):
    def __init__():
        self.W = np.eye()

    def energy(self, state):
        pass

class RBoltzmann1(DistrInterface):
    def get_dual(self):
        return self.dual
    def energy(self, state):
        (v, h) = state
        pass

class BinaryDataProvider(object):
    def get_sample(self, i):
        return None

    def samples(self):
        return 0

    def get_next_sample(self):
        yield None

    def shuffle(self):
        """ prepare for next shuffled"""
        pass

    def get_next_shuffled(self):
        pass

    def set_mode(self, mode):
        assert mode in ['train', 'test', 'validation']

    def format(self, sample_vector):
        pass

#http://deeplearning.net/datasets/
#class test_

class MNISTLoader(BinaryDataProvider):
    preloaded = False
    @staticmethod
    def preload(path):
        import os
        #path = '/home/sohail/ml/datasets/mnist'
        absolute_filename = os.path.join(path, 'mnist.pkl.gz')

        if VERBOSE:
            print 'loading MNIST', ;flush_stdout()
        # Loading code from: http://deeplearning.net/tutorial/gettingstarted.html
        import cPickle, gzip, numpy
        f = gzip.open(absolute_filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        if VERBOSE:
            print 'done.' ;flush_stdout()

        MNISTLoader.train_set, MNISTLoader.valid_set, MNISTLoader.test_set = train_set, valid_set, test_set

    def __init__(self, mode='train'):
        if not MNISTLoader.preloaded:
            # Downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz
            MNISTLoader.preload('/home/sohail/ml/datasets/mnist')
            MNISTLoader.preloaded = True
        self.set_mode(mode)

    def set_mode(self, mode):
        assert mode in ['train', 'test', 'validation']
        mode__dset_lookup = {'train': 0, 'test': 1, 'validation': 2}
        datasets = [MNISTLoader.train_set, MNISTLoader.valid_set, MNISTLoader.test_set]
        self.active_dataset = datasets[mode__dset_lookup[mode]]
        if VERBOSE:
            print 'active dataset: \'%s\''%(mode,)

    def format(self, sample_vector):
        return sample_vector.reshape(28, 28)

def flush_stdout():
    import sys
    sys.stdout.flush()

def test_mnist():
    #print 'loading', ;flush_stdout()
    d = MNISTLoader()
    #print 'done.' ;flush_stdout()
    print d
    print type(d.train_set)
    print len(d.train_set)
    print d.train_set

    print d.train_set[0].shape #(50000, 784)
    print d.train_set[1].shape #(50000,) of int64  #labels

    print d.train_set[1][0]
    print type(d.train_set[1][0])  # int64

    print d.train_set[0][0,0]
    print type(d.train_set[0][0,0])  # float32



    for t in [MNISTLoader.train_set, MNISTLoader.valid_set, MNISTLoader.test_set]:
        print t[1].shape,  #labels
        print t[0].shape,  #data
        print type(t[1][0]),  # int64
        print type(t[0][0,0])  # float32
        #(50000,) (50000, 784) <type 'numpy.int64'> <type 'numpy.float32'>
        #(10000,) (10000, 784) <type 'numpy.int64'> <type 'numpy.float32'>
        #(10000,) (10000, 784) <type 'numpy.int64'> <type 'numpy.float32'>

        def print_image(image):
            #exit()
            w = 28
            for y in range(784/w):
                for x in range(w):
                    print '.' if image[x+w*y] < 0.5 else '1',
                print
            print

        vector = t[0]
        print (np.min(vector), np.max(vector)),  #(0,0.996)
        i = 100
        image = vector[i]
        print_image(image)
        
        #matrix = image.reshape(28, 28)
        #print np.floor(matrix*10).astype(int)

def factorize():    
        print 7.*8.*7.*2
        m = 784./7./8./7./2.
        print m, ":",
        for i in range(2, int(m**0.5)):
            if float(m)/float(i) % 1. == 0.:
                print i,
        print  # 2 4 7 8 14 16

if __name__ == '__main__':
    #test_mnist()
    #factorize()
    d = MNISTLoader('test')
    vec = d.get_samepl(1)
