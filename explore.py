import cPickle
import numpy as np
import matplotlib.pyplot as plt
import theano


filename1 = '/data/lisatmp3/shahamar/2015-11-03-adding-750-fft-5scalepen.pkl'
#filename1 = '/data/lisatmp3/shahamar/2015-11-01-adding-500-fft-5scalepen.pkl'

f1 = file(filename1, 'rb')
data1 = cPickle.load(f1)
f1.close()

import pdb; pdb.set_trace()

name = 'test_loss'

loss1 = data1[name]








plt.plot(loss1, 'g')
med = np.median(data1[name][100:])
plt.plot(med*np.ones_like(data1[name]))

plt.show()
