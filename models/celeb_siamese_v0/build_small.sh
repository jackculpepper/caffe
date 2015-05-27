
NUM_EXAMPLES=500
NUM_EXAMPLES=1000
NUM_EXAMPLES=10000

head -${NUM_EXAMPLES} clusters_2_seed=0_trn_siamese.txt >clusters_2_seed=0_trn_siamese_small.txt
head -${NUM_EXAMPLES} clusters_2_seed=0_trn_siamese_b.txt >clusters_2_seed=0_trn_siamese_small_b.txt
head -${NUM_EXAMPLES} clusters_2_seed=0_tst_siamese.txt >clusters_2_seed=0_tst_siamese_small.txt
head -${NUM_EXAMPLES} clusters_2_seed=0_tst_siamese_b.txt >clusters_2_seed=0_tst_siamese_small_b.txt

