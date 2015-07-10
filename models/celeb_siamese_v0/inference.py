
import caffe
import numpy as np
import json
from PIL import Image
from time import time
from cStringIO import StringIO

from os.path import join
import sys

from collections import defaultdict

np.random.seed(0)


from argparse import ArgumentParser
parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)


model_root = '.'

model_file = 'celeb_v23_iter_2370000.caffemodel'
model_file = 'celeb_v17_iter_550000.caffemodel'
model_file = 'celeb_v23_iter_2800000.caffemodel'


model_prototxt = 'test_siamese_memorydata.prototxt'
model_prototxt = 'test_feature.prototxt'

mean_file = 'celeb34372_mean.npy'
mean_file = 'celeb34372_mean.binaryproto'


lfw_a_file = join(model_root, 'lfw.txt')
lfw_b_file = join(model_root, 'lfw_b.txt')

lfw_a_file = join(model_root, 'lfw_v2_a.txt')
lfw_b_file = join(model_root, 'lfw_v2_b.txt')

lfw_knn_file = join(model_root, 'lfw_v2_knn.txt')
tobi_file = join(model_root, 'tobi.txt')

lines_cache = {}
images_cache = {}
images_cache_hits = 0


def load_mean():
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    return arr


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


def load_batch(data_file, mean_trn_img,
               batch_num, batch_size=50, transpose=(2,0,1),
               channel_swap=(2,1,0), verbose=False):
    t0 = time()

    lines = load_lines(data_file)
    global images_cache_hits
    images_cache_hits = 0

    data = np.zeros((batch_size, 3, 227, 227), dtype=np.float32)
    labels = np.zeros((batch_size, 1, 1, 1), dtype=np.float32)
    paths = list()

    for i in range(batch_size):
        if batch_num == -1:
            j = np.random.randint(len(lines))
        else:
            j = batch_num * batch_size + i
            if j > len(lines):
                j = j % len(lines)

        image_path, label = lines[j].split()
        paths.append(image_path)
        label = float(label)

        input_image = load_image(image_path)
        input_image_t = input_image.transpose(transpose)
        input_image_t_swap = input_image_t[channel_swap, :, :]

        data[i, :, :, :] = input_image_t_swap - mean_trn_img
        labels[i, 0, 0, 0] = label

    if verbose:
        print 'loaded batch in %.3f s' % ( time() - t0 ),
        print '%d cache hits' % images_cache_hits

    return data, labels, paths





def load_lfw_a_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(lfw_a_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def load_lfw_b_batch(batch_num, batch_size, mean_trn_img):
    return load_batch(lfw_b_file, mean_trn_img,
                      batch_num, batch_size=batch_size)

def run_batches(net, num_tst_batches, batch_size, mean_trn_img):
    dists = list()
    same_labels = list()

    for i in range(num_tst_batches):
        print '%d / %d' % (i, num_tst_batches)

        data_lfw_a, labels_lfw_a, paths_lfw_a = load_batch(lfw_a_file, mean_trn_img, i, batch_size)
        data_lfw_b, labels_lfw_b, paths_lfw_b = load_batch(lfw_b_file, mean_trn_img, i, batch_size)

        net.set_input_arrays('data',   data_lfw_a, labels_lfw_a)
        net.set_input_arrays('data_b', data_lfw_b, labels_lfw_b)

        net.forward()
        same_label = np.array(net.blobs['same_label'].data.squeeze())
        #ip2_a = net.blobs['ip2'].data.squeeze()
        #ip2_b = net.blobs['ip2_b'].data.squeeze()
        feat_a = net.blobs['fc6_celeb'].data.squeeze()
        feat_b = net.blobs['fc6_celeb_b'].data.squeeze()

        diff = feat_a - feat_b
        dist = (diff**2).sum(axis=1)

        same_labels.append(same_label)
        dists.append(dist)

    dists = np.hstack(dists)
    same_labels = np.hstack(same_labels)

    return dists, same_labels


def run_knn(net, num_tst_batches, batch_size, mean_trn_img, top_k=10):
    features = list()
    labels = list()
    paths = list()

    knn_file = lfw_knn_file
    knn_file = tobi_file

    for i in range(num_tst_batches):
        print '%d / %d' % (i, num_tst_batches)

        data_lfw, labels_lfw, paths_lfw = load_batch(knn_file, mean_trn_img, i, batch_size)

        net.set_input_arrays('data', data_lfw, labels_lfw)

        net.forward()
        #ip2_a = net.blobs['ip2'].data.squeeze()
        feat = net.blobs['fc6_celeb'].data.squeeze()

        features.append(np.array(feat))
        labels.append(np.array(labels_lfw.squeeze()))
        paths.extend(paths_lfw)


    features_mat = np.vstack(features)
    labels_mat = np.hstack(labels)

    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform

    dists = squareform(pdist(features_mat, 'euclidean'))
    dists_sorted_idx = np.argsort(dists, 0)

    import IPython ; IPython.embed()

    d = dict()
    d['features'] = features_mat
    d['paths'] = paths

    with open('feats.pkl', 'w') as fh:
        pickle.dump(d, fh)

    out_dir = '/raid/tobi/data/account_faces/ranked'

    for i in range(dists.shape[0]):
        q = paths[dists_sorted_idx[0,i]]
        print 'query', q

        fname = os.path.basename(q)
        out_file = join(out_dir, fname + '___.jpg')
        shutil.copy(q, out_file)

        for j in range(top_k):
            p = paths[dists_sorted_idx[j,i]]
            print p

            out_file = fname + "_%02d_" % j + os.path.basename(p)

            out_file = join(out_dir, out_file)

            shutil.copy(p, out_file)

        import IPython ; IPython.embed()



def calc_acc(dists, same_labels, thresh):
    same_idx = np.where(same_labels == 1)
    diff_idx = np.where(same_labels == 0)

    num_right = 0.0
    num_right += (dists[same_idx] < thresh).sum()
    num_right += (dists[diff_idx] >= thresh).sum()

    acc = num_right / len(same_labels)

    return acc

def sweep_thresh(dists, same_labels):
    thresh_list = list(dists)
    thresh_list.sort()

    same_idx = np.where(same_labels == 1)
    diff_idx = np.where(same_labels == 0)


    max_acc = 0.0
    max_thresh = -1

    for thresh in thresh_list:
        acc = calc_acc(dists, same_labels, thresh)

        print 'thresh', thresh, 'acc', acc

        if acc > max_acc:
            max_acc = acc
            max_thresh = thresh

    return max_acc, max_thresh

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        device_id = int(args.device_id)
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()

    net = caffe.Net(model_prototxt, model_file, caffe.TEST)

    batch_size = net.blobs['data'].data.shape[0]
    batch_shape = net.blobs['data'].data.shape[1:]

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in net.blobs.items()]

    # just print the weight sizes (not biases)
    print [(k, v[0].data.shape) for k, v in net.params.items()]


    mean_trn_img = load_mean()

    if True:
        num_tst_batches = 24
        num_tst_batches = 426
        run_knn(net, num_tst_batches, batch_size, mean_trn_img)



    if False:
        # contrastive loss for first batch: 0.06643287092447281

        num_tst_batches = 24
        dists, same_labels = run_batches(net, num_tst_batches, batch_size, mean_trn_img)

        max_acc, max_thresh = sweep_thresh(dists, same_labels)

        print 'max_thresh', max_thresh, 'max_acc', max_acc

        num_tst_batches = 96
        num_tst_batches = 237
        num_tst_batches = 240

        dists, same_labels = run_batches(net, num_tst_batches, batch_size, mean_trn_img)

        acc = calc_acc(dists, same_labels, max_thresh)

        print 'thresh', max_thresh, 'acc', acc




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


