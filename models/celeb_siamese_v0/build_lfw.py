

import os

lfw_list = '/nfs/jack/data/lfw/lfw_list.txt'
pairs_path = '/nfs/jack/data/lfw/pairs.txt'
img_root = '/raid/jack/data/lfw_orig.balanced.faces.v2,size=227,padding=0.3,upsample=0'

with open(lfw_list, 'r') as fh: lfw_names = fh.readlines()

lfw_names = [ l.strip() for l in lfw_names ]
#lfw_names = [ ''.join(l.split('_')) for l in lfw_names ]

#import IPython; IPython.embed()

name_to_int = dict()
for i, name in enumerate(lfw_names):
    name_to_int[name] = i

lfw_a_fh = open('lfw_a_v2.txt', 'w')
lfw_b_fh = open('lfw_b_v2.txt', 'w')

with open(pairs_path, 'r') as fh:
    splits, n_per_split = fh.readline().strip().split()
    splits, n_per_split = int(splits), int(n_per_split)

    for i in range(splits):
        # positive examples
        for j in range(n_per_split):
            name, a, b = fh.readline().strip().split()
            a, b = int(a), int(b)
            #print name, a, name, b

            sdir = os.path.join(name[:2], name)
            fname_a = '%s_%04d.jpg.0.jpg' % (name, a)
            fname_b = '%s_%04d.jpg.0.jpg' % (name, b)

            dirname = os.path.join(img_root, sdir)
            path_a = os.path.join(dirname, fname_a)
            path_b = os.path.join(dirname, fname_b)

            if os.path.isfile(path_a):
                if os.path.isfile(path_b):
                    lfw_a_fh.write(path_a + ' %d\n' % name_to_int[name])
                    lfw_b_fh.write(path_b + ' %d\n' % name_to_int[name])
                else:
                    print 'missing', path_b
            else:
                print 'missing', path_a

        # negative examples
        for j in range(n_per_split):
            name_a, a, name_b, b = fh.readline().strip().split()

            a, b = int(a), int(b)
            #print name_a, a, name_b, b

            sdir_a = os.path.join(name_a[:2], name_a)
            fname_a = '%s_%04d.jpg.0.jpg' % (name_a, a)
            dirname_a = os.path.join(img_root, sdir_a)
            path_a = os.path.join(dirname_a, fname_a)

            sdir_b = os.path.join(name_b[:2], name_b)
            fname_b = '%s_%04d.jpg.0.jpg' % (name_b, b)
            dirname_b = os.path.join(img_root, sdir_b)
            path_b = os.path.join(dirname_b, fname_b)

            if os.path.isfile(path_a):
                if os.path.isfile(path_b):
                    lfw_a_fh.write(path_a + ' %d\n' % name_to_int[name_a])
                    lfw_b_fh.write(path_b + ' %d\n' % name_to_int[name_b])
                else:
                    print 'missing', path_b
            else:
                print 'missing', path_a

