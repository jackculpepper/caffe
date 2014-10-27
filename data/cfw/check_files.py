
fh_a = open('cfw_train_siamese_a.txt', 'r')
fh_b = open('cfw_train_siamese_b.txt', 'r')

count_same = 0
count_notsame = 0

for line_a in fh_a:
    line_b = fh_b.readline()
    line_b.strip()

    label_a = line_a.split()[1]
    label_b = line_b.split()[1]

    if label_a == label_b:
        count_same += 1
    else:
        count_notsame += 1

print count_same, count_notsame
    
        
