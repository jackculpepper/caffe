
from collections import defaultdict
import random

# truncate same pairs
max_same_per_label = 1000

tt = 'test'
tt = 'train'

label_to_images = defaultdict(list)

with open('cfw_%s_random_a.txt' % tt, 'r') as fh:
    for line in fh:
        path, label = line.split()
        label = int(label)
        label_to_images[label].append(path)

labels = label_to_images.keys()
labels.sort()

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

    print '\r%d / %d' % (label, len(labels)),
print

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

    print '\r%d / %d' % (len(not_same_pairs), num_same_pairs),
print

# output pairs
pairs = list()
pairs.extend(same_pairs)
pairs.extend(not_same_pairs)
random.shuffle(pairs)

fh_a = open('cfw_%s_siamese_a.txt' % tt, 'w')
fh_b = open('cfw_%s_siamese_b.txt' % tt, 'w')

for p in pairs:
    label_a, label_b, image_a, image_b = p
    path_a = label_to_images[label_a][image_a]
    path_b = label_to_images[label_b][image_b]
    fh_a.write(path_a + ' ' + str(label_a) + '\n')
    fh_b.write(path_b + ' ' + str(label_b) + '\n')

fh_a.close()
fh_b.close()

