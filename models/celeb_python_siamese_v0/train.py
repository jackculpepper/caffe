
import caffe
import numpy as np
from PIL import Image
from time import time
from cStringIO import StringIO

import sys
import os

from os.path import join
from collections import defaultdict

np.random.seed(0)



from argparse import ArgumentParser
parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)


model_root = 'models/celeb_python_siamese_v0'

mean_file = join(model_root, 'celeb_mean.npy')

trn_src_file = join(model_root, 'clusters_2_seed=0_trn.txt.shuf' % tt, 'r')


trn_file = join(model_root, 'clusters_2_seed=0_trn_siamese.txt')
trn_b_file = join(model_root, 'clusters_2_seed=0_trn_siamese_b.txt')

tst_file = join(model_root, 'clusters_2_seed=0_tst_siamese.txt')
tst_b_file = join(model_root, 'clusters_2_seed=0_tst_siamese_b.txt')

#tst_bng_file = join(model_root, 'binger_clean_tst.txt.shuf')
#tst_bng_b_file = join(model_root, 'binger_clean_tst_b.txt.shuf')

lfw_file = join(model_root, 'lfw.txt')
lfw_b_file = join(model_root, 'lfw_b.txt')

solver_file = join(model_root, 'solver.prototxt')
#model_file = join(model_root, 'celeb_v0_iter_80000.caffemodel')
model_file = join(model_root, 'celeb_v1_iter_280000.caffemodel')
#model_file = join(model_root, 'celeb_v0_iter_10000.caffemodel')

lines_cache = {}
images_cache = {}
images_cache_hits = 0


label_to_images_trn = None
label_to_images_tst = None


def generate_pairs(labels, label_to_images):

    # same pairs
    same_pairs = list()

    num_same_pairs = 0

    # generate same pairs
    for label in labels:
        same_pairs_for_label = list()

        paths = label_to_images[label]
        k = 0
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                if k % 2 == 0:
                    tup = (i, j)
                else:
                    tup = (j, i)

                tup = (label, label) + tup
                same_pairs_for_label.append(tup)
                k += 1

        # shuffle and truncate
        random.shuffle(same_pairs_for_label)
        same_pairs_for_label = same_pairs_for_label[:max_same_per_label]

        num_same_pairs += len(same_pairs_for_label)
        same_pairs.extend(same_pairs_for_label)

    # not same pairs
    not_same_pairs = set()

    while len(not_same_pairs) < num_same_pairs:
        # pick labels
        label_a = label_b = 0
        while label_a == label_b:
            label_a = random.randint(0, len(labels) - 1)
            label_b = random.randint(0, len(labels) - 1)

        # pick images
        image_a = random.randint(0, len(label_to_images[label_a]) - 1)
        image_b = random.randint(0, len(label_to_images[label_b]) - 1)

        tup = (label_a, label_b, image_a, image_b)
        not_same_pairs.add(tup)






def build_label_map(data_file):
    label_to_images = defaultdict(list)

    with open(data_file, 'r') as fh:
        for line in fh:
            path, label = line.split()
            label = int(label)
            label_to_images[label].append(path)

    labels = label_to_images.keys()
    labels.sort()

    return label_to_images, labels


def load_classifier():
    net = caffe.Classifier(model_prototxt, model_file,
                           mean=np.load(mean_file),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(227, 227))
    return net


def compute_mean_image(data_file, num_images=100000):
    lines = load_lines(data_file)
    sum = None
    den = 0

    for i, l in enumerate(lines):
        if i >= num_images: break

        image_path, label = l.split()
        print 'loading %d / %d : %s' % (i+1, num_images, image_path)
        input_image = load_image(image_path)
        input_image_t = input_image.transpose(transpose)

        if sum is None: sum = input_image_t
        else: sum += input_image_t

        den += 1


    return sum / den







def load_image(img_path):
    t0 = time()
    global images_cache_hits

    if not images_cache.has_key(img_path):
        img_data = open(img_path, 'r').read()
        images_cache[img_path] = img_data
    else:
        img_data = images_cache[img_path]
        images_cache_hits += 1
    img_io = StringIO(img_data)

    # open the image
    try:
        img = Image.open(img_io)
        width, height = img.size
    except IOError:
        raise IOError("broken image (cannot open)")

    # check supported format
    supported_formats = ['JPEG', 'GIF', 'PNG', 'BMP', 'TIFF']
    if not img.format in supported_formats:
        raise NotImplementedError('unknown image format %s' % img.format)

    # check supported mode
    supported_modes = ['RGB', 'RGBA', 'CMYK', 'P', 'L', 'LA', '1']
    if not img.mode in supported_modes:
        raise NotImplementedError('unknown image mode %s' % img.mode)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    pixels = np.array(img, dtype=np.float32)

    #print 'loaded image in %.3f s' % ( time() - t0 )

    return pixels





def load_lines(data_file):
    if not lines_cache.has_key(data_file):
        lines = list()
        with open(data_file, 'r') as fh:
            lines = fh.readlines()
            lines = [ l.strip() for l in lines ]
        lines_cache[data_file] = lines
    else:
        lines = lines_cache[data_file]

    return lines

def load_batch(data_file, mean_trn_img,
               batch_num, batch_size=50, transpose=(2,0,1), verbose=False):
    t0 = time()

    lines = load_lines(data_file)
    global images_cache_hits
    images_cache_hits = 0

    data = np.zeros((batch_size, 3, 227, 227), dtype=np.float32)
    labels = np.zeros((batch_size, 1, 1, 1), dtype=np.float32)

    for i in range(batch_size):
        if batch_num == -1:
            j = np.random.randint(len(lines))
        else:
            j = batch_num * batch_size + i
            if j > len(lines):
                j = j % len(lines)

        image_path, label = lines[j].split()
        label = float(label)

        input_image = load_image(image_path)
        input_image_t = input_image.transpose(transpose)

        data[i, :, :, :] = input_image_t - mean_trn_img
        labels[i, 0, 0, 0] = label

    if verbose:
        print 'loaded batch in %.3f s' % ( time() - t0 ),
        print '%d cache hits' % images_cache_hits

    return data, labels

def load_tst_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(tst_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_tst_b_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(tst_b_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_lfw_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(lfw_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_lfw_b_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(lfw_b_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_trn_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(trn_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_trn_b_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(trn_b_file, mean_trn_img,
                      batch_num, batch_size=batch_size)


def warm(data_file):
    lines = load_lines(data_file)
    for i, l in enumerate(lines):
        image_path, label = l.split()
        print 'warming %d / %d : %s' % (i+1, len(lines), image_path)
        input_image = load_image(image_path)


def compute_mean_image(data_file, num_images=100000, transpose=(2,0,1)):
    lines = load_lines(data_file)
    sum = None
    den = 0

    for i, l in enumerate(lines):
        if i >= num_images: break

        image_path, label = l.split()
        print 'loading %d / %d : %s' % (i+1, num_images, image_path)
        input_image = load_image(image_path)
        input_image_t = input_image.transpose(transpose)

        if sum is None: sum = input_image_t
        else: sum += input_image_t

        den += 1


    return sum / den


def display_data():
    # we use a little trick to tile the first eight images
    #imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
    print solver.net.blobs['label'].data[:8]





def test_sgd(solver, mean_trn_img, batch_size, niter=10, num_test_batches=10):
    # connect test network memory data layers to python managed buffers
    data_tst, labels_tst = load_tst_batch(0, batch_size, mean_trn_img)
    solver.test_nets[1].set_input_arrays('data', data_tst, labels_tst)

    data_tst_b, labels_tst_b = load_tst_b_batch(0, batch_size, mean_trn_img)
    solver.test_nets[1].set_input_arrays('data_b', data_tst_b, labels_tst_b)

    data_lfw, labels_lfw = load_lfw_batch(0, batch_size, mean_trn_img)
    solver.test_nets[0].set_input_arrays('data', data_lfw, labels_lfw)

    data_lfw_b, labels_lfw_b = load_lfw_b_batch(0, batch_size, mean_trn_img)
    solver.test_nets[0].set_input_arrays('data_b', data_lfw_b, labels_lfw_b)



    trn_kway_acc = np.zeros(int(np.ceil(niter)))
    trn_pair_acc = np.zeros(int(np.ceil(niter)))
    tst_kway_acc = np.zeros(int(np.ceil(niter)))
    tst_pair_acc = np.zeros(int(np.ceil(niter)))
    lfw_pair_acc = np.zeros(int(np.ceil(niter)))


    num_trn_batches = len(load_lines(trn_file)) / batch_size


    for it in range(niter):

        for i in range(num_test_batches):
            batch_num = np.random.randint(num_trn_batches)

            data_trn, labels_trn = load_trn_batch(batch_num, batch_size, mean_trn_img)
            solver.net.set_input_arrays('data', data_trn, labels_trn)

            data_trn_b, labels_trn_b = load_trn_b_batch(batch_num, batch_size, mean_trn_img)
            solver.net.set_input_arrays('data_b', data_trn_b, labels_trn_b)

            solver.step(1)

        # trn kway and pair classification
        kway_correct = 0.0
        pair_correct = 0.0
        for i in range(num_test_batches):

            data_trn, labels_trn = load_trn_batch(i, batch_size, mean_trn_img)
            solver.net.set_input_arrays('data', data_trn, labels_trn)

            data_trn_b, labels_trn_b = load_trn_b_batch(i, batch_size, mean_trn_img)
            solver.net.set_input_arrays('data_b', data_trn_b, labels_trn_b)

            solver.net.forward()

            kway_correct += (solver.net.blobs['fc7'].data.argmax(1)
                          == solver.net.blobs['label'].data).sum()
            pair_correct += (solver.net.blobs['ip3'].data.argmax(1)
                          == solver.net.blobs['same_label'].data.squeeze()).sum()

        trn_kway_acc[it] = kway_correct / (batch_size * num_test_batches)
        trn_pair_acc[it] = pair_correct / (batch_size * num_test_batches)

        # tst kway and pair classification
        kway_correct = 0.0
        pair_correct = 0.0
        for i in range(num_test_batches):

            data_tst, labels_tst = load_tst_batch(i, batch_size, mean_trn_img)
            solver.test_nets[1].set_input_arrays('data', data_tst, labels_tst)

            data_tst_b, labels_tst_b = load_tst_b_batch(i, batch_size, mean_trn_img)
            solver.test_nets[1].set_input_arrays('data_b', data_tst_b, labels_tst_b)

            solver.test_nets[1].forward()


            kway_correct += (solver.test_nets[1].blobs['fc7'].data.argmax(1)
                          == solver.test_nets[1].blobs['label'].data).sum()
            pair_correct += (solver.test_nets[1].blobs['ip3'].data.argmax(1)
                          == solver.test_nets[1].blobs['same_label'].data.squeeze()).sum()

        tst_kway_acc[it] = kway_correct / (batch_size * num_test_batches)
        tst_pair_acc[it] = pair_correct / (batch_size * num_test_batches)


        # lfw classification
        pair_correct = 0.0
        for i in range(num_test_batches):

            data_lfw, labels_lfw = load_lfw_batch(i, batch_size, mean_trn_img)
            solver.test_nets[0].set_input_arrays('data', data_lfw, labels_lfw)

            data_lfw_b, labels_lfw_b = load_lfw_b_batch(i, batch_size, mean_trn_img)
            solver.test_nets[0].set_input_arrays('data_b', data_lfw_b, labels_lfw_b)

            solver.test_nets[0].forward()

            pair_correct += (solver.test_nets[0].blobs['ip3'].data.argmax(1)
                           == solver.test_nets[0].blobs['same_label'].data.squeeze()).sum()


        lfw_pair_acc[it] = pair_correct / (batch_size * num_test_batches)

        print 'trn_kway_acc', trn_kway_acc[it]
        print 'trn_pair_acc', trn_pair_acc[it]
        print 'tst_kway_acc', tst_kway_acc[it]
        print 'tst_pair_acc', tst_pair_acc[it]
        print 'lfw_pair_acc', lfw_pair_acc[it]





def sgd(solver, mean_trn_img, batch_size):

    niter = 280000
    test_interval = 1000
    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    trn_kway_acc = np.zeros(int(np.ceil(niter / test_interval)))
    trn_pair_acc = np.zeros(int(np.ceil(niter / test_interval)))
    tst_kway_acc = np.zeros(int(np.ceil(niter / test_interval)))
    tst_pair_acc = np.zeros(int(np.ceil(niter / test_interval)))
    lfw_pair_acc = np.zeros(int(np.ceil(niter / test_interval)))
    output = np.zeros((niter, 8, 10))

    num_test_batches = 20
    num_test_batches = 118

    #import IPython ; IPython.embed()


    # connect test network memory data layers to python managed buffers
    data_tst, labels_tst = load_tst_batch(0, batch_size, mean_trn_img)
    solver.test_nets[1].set_input_arrays('data', data_tst, labels_tst)

    data_tst_b, labels_tst_b = load_tst_b_batch(0, batch_size, mean_trn_img)
    solver.test_nets[1].set_input_arrays('data_b', data_tst_b, labels_tst_b)

    data_lfw, labels_lfw = load_lfw_batch(0, batch_size, mean_trn_img)
    solver.test_nets[0].set_input_arrays('data', data_lfw, labels_lfw)

    data_lfw_b, labels_lfw_b = load_lfw_b_batch(0, batch_size, mean_trn_img)
    solver.test_nets[0].set_input_arrays('data_b', data_lfw_b, labels_lfw_b)




    # the main solver loop
    for it in range(niter):

        batch_num = np.random.randint(num_test_batches)

        data_trn, labels_trn = load_trn_batch(batch_num, batch_size, mean_trn_img)
        solver.net.set_input_arrays('data', data_trn, labels_trn)
        data_trn_b, labels_trn_b = load_trn_b_batch(batch_num, batch_size, mean_trn_img)
        solver.net.set_input_arrays('data_b', data_trn_b, labels_trn_b)

        solver.step(1)  # SGD by Caffe
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
    
        # store the output on the first test batch
        #solver.test_nets[0].forward()
        #solver.test_nets[1].forward()
        #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
    
        # run a full test every so often
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            test_it = it // test_interval

            # trn kway and pair classification
            kway_correct = 0.0
            pair_correct = 0.0
            for i in range(num_test_batches):

                data_trn, labels_trn = load_trn_batch(i, batch_size, mean_trn_img)
                solver.net.set_input_arrays('data', data_trn, labels_trn)

                data_trn_b, labels_trn_b = load_trn_b_batch(i, batch_size, mean_trn_img)
                solver.net.set_input_arrays('data_b', data_trn_b, labels_trn_b)

                solver.net.forward()

                kway_correct += (solver.net.blobs['fc7'].data.argmax(1)
                              == solver.net.blobs['label'].data).sum()
                pair_correct += (solver.net.blobs['ip3'].data.argmax(1)
                              == solver.net.blobs['same_label'].data.squeeze()).sum()

            trn_kway_acc[test_it] = kway_correct / (batch_size * num_test_batches)
            trn_pair_acc[test_it] = pair_correct / (batch_size * num_test_batches)


            # tst kway and pair classification
            kway_correct = 0.0
            pair_correct = 0.0
            for i in range(num_test_batches):

                data_tst, labels_tst = load_tst_batch(i, batch_size, mean_trn_img)
                solver.test_nets[1].set_input_arrays('data', data_tst, labels_tst)

                data_tst_b, labels_tst_b = load_tst_b_batch(i, batch_size, mean_trn_img)
                solver.test_nets[1].set_input_arrays('data_b', data_tst_b, labels_tst_b)

                solver.test_nets[1].forward()


                kway_correct += (solver.test_nets[1].blobs['fc7'].data.argmax(1)
                              == solver.test_nets[1].blobs['label'].data).sum()
                pair_correct += (solver.test_nets[1].blobs['ip3'].data.argmax(1)
                              == solver.test_nets[1].blobs['same_label'].data.squeeze()).sum()

            tst_kway_acc[test_it] = kway_correct / (batch_size * num_test_batches)
            tst_pair_acc[test_it] = pair_correct / (batch_size * num_test_batches)


            # lfw classification
            pair_correct = 0.0
            for i in range(num_test_batches):

                data_lfw, labels_lfw = load_lfw_batch(i, batch_size, mean_trn_img)
                solver.test_nets[0].set_input_arrays('data', data_lfw, labels_lfw)

                data_lfw_b, labels_lfw_b = load_lfw_b_batch(i, batch_size, mean_trn_img)
                solver.test_nets[0].set_input_arrays('data_b', data_lfw_b, labels_lfw_b)

                solver.test_nets[0].forward()

                pair_correct += (solver.test_nets[0].blobs['ip3'].data.argmax(1)
                               == solver.test_nets[0].blobs['same_label'].data.squeeze()).sum()

                #import IPython ; IPython.embed()

            lfw_pair_acc[test_it] = pair_correct / (batch_size * num_test_batches)


            print 'trn_kway_acc', trn_kway_acc[test_it]
            print 'trn_pair_acc', trn_pair_acc[test_it]
            print 'tst_kway_acc', tst_kway_acc[test_it]
            print 'tst_pair_acc', tst_pair_acc[test_it]
            print 'lfw_pair_acc', lfw_pair_acc[test_it]


def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        device_id = int(args.device_id)
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()


    solver = caffe.SGDSolver(solver_file)
    solver.net.copy_from(model_file)

    batch_size = solver.net.blobs['data'].data.shape[0]
    batch_shape = solver.net.blobs['data'].data.shape[1:]

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

    # just print the weight sizes (not biases)
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]


    if os.path.isfile(mean_file):
        mean_trn_img = np.load(mean_file)
        print 'loaded mean training image'
    else:
        mean_trn_img = compute_mean_image(trn_file)
        print 'calculated mean training image'
        np.save(mean_file, mean_trn_img)
    
    if False:
        warm(trn_file)


    # prime test network data layers

    #test_sgd(solver, mean_trn_img, batch_size, niter=10, num_test_batches=10)
    #import IPython ; IPython.embed()

    sgd(solver, mean_trn_img, batch_size)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

