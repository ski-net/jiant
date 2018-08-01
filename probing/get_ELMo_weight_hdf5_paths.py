import h5py

f = h5py.File('/nfs/jsalt/share/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', 'r')
l = lambda x: print(x)
f['/'].visit(l)
