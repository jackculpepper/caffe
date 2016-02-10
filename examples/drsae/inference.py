
import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from os.path import join
import IPython
from argparse import ArgumentParser

np.random.seed(0)

parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)
parser.add_argument('-c', '--checkpoint', type=int, help='''checkpoint number''',
                    default=-1)

root_dir = 'examples/drsae'
model_file = 'drsae_v00_iter_%d.caffemodel'
model_file = 'drsae_v01_iter_%d.caffemodel'
model_file = 'drsae_v02_iter_%d.caffemodel'
model_file = 'drsae_v03_iter_%d.caffemodel'
model_prototxt = 'train_test.prototxt'

def vis_square(data, padsize=1, padval=0):
    # maximize contrast
    data -= data.min()
    data /= data.max()
    # balance zero at gray
    #data /= np.abs(data).max()
    #data += 1
    #data /= 2.0
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        print 'running on gpu %d' % args.device_id
        caffe.set_device(args.device_id)
        caffe.set_mode_gpu()

    c = args.checkpoint

    #batch_size  = net.blobs['data'].data.shape[0]
    #batch_shape = net.blobs['data'].data.shape[1:]


    #input_dat = np.squeeze(np.array(net.blobs['data'].data))
    #input_vis = vis_square(input_dat)
    #input_img = np.uint8(input_vis*255)
    #Image.fromarray(input_img).save(join(root_dir, 'input_img.png'))


    net = caffe.Net(join(root_dir, model_prototxt),
                    join(root_dir, model_file % c), caffe.TEST)

    if False:
        net.forward()
        IPython.embed()

        for i in range(2, 41, 2):
            coefs = net.blobs['z_%03d' % i].data
            sparse_loss = net.blobs['sparse_loss_%03d' % i].data
            euclidean_loss = net.blobs['euclidean_loss_%03d' % i].data
            recon = net.blobs['recon_%03d' % i].data
            recon_vis = vis_square(recon.reshape(100,28,28))
            recon_img = np.uint8(recon_vis*255)
            Image.fromarray(recon_img).save(join(root_dir, 'recon_img_layer_%03d_iter_%d.png' % (i, c)))


    w_enc = net.params['Ex'][0].data
    b_enc = net.params['Ex'][1].data
    w_enc_vis = vis_square(w_enc.reshape(400,28,28))
    w_enc_img = np.uint8(w_enc_vis*255)
    Image.fromarray(w_enc_img).save(join(root_dir, 'w_enc_img_iter_%d.png' % c))

    w_dec = net.params['recon_002'][0].data
    b_dec = net.params['recon_002'][1].data
    w_dec_vis = vis_square(w_dec.T.reshape(400,28,28))
    w_dec_img = np.uint8(w_dec_vis*255)
    Image.fromarray(w_dec_img).save(join(root_dir, 'w_dec_img_iter_%d.png' % c))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


