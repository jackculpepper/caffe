#!/usr/bin/env python

import os, sys, glob, shutil, socket

sys.path.append('/home-2/jculpepper/caffe/python')
sys.path.append('/home-2/jculpepper/vision-hype')
import caffe

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from hype import GPUWorkr
from tempfile import mkstemp


def parse_score(f, debug=False):
    lines = f.readlines()
    score = 0.0

    l = 0
    while l < len(lines):
        if "Testing net (#0)" in lines[l]:
            iter = lines[l].split()[-4][:-1]
            output = dict()
            output['lfw/accuracy'] = float(lines[l + 1].split()[-1])

            # jump to the next test network
            l += 3
            assert("Testing net (#1)" in lines[l])

            # jump to the siamese accuracy
            l += 5
            assert("Test net output #4: contrast/siamese_accuracy" in lines[l])
            output['cfw_test/accuracy'] = float(lines[l].split()[-1])

            # check to see if we have improved on the validation set
            if output['cfw_test/accuracy'] > score:
                # we have; record test accuracy
                score = output['lfw/accuracy']

                if debug:
                    print 'best', iter,
                    print 'val', output['cfw_test/accuracy'],
                    print 'test', output['lfw/accuracy']

            # jump to the last test output line
            l += 1

        # increment line
        l += 1

    return score

def test_parse_score():
    p = 'caffe.ivb113.jculpepper.log.INFO.20141114-190712.30159'
    with open(p, 'r') as fh:
        parse_score(fh)


def main():
    folder = sys.argv[1]
    experiment = sys.argv[2]
    device = int(sys.argv[3])

    w = FaceWorkr(folder, experiment, device)
    w.loop()

class FaceWorkr(GPUWorkr):
    def gpu_run(self, job, output0, output1, output2, btlneck, loss_weights, stepsize, base_lr):
        log = self.folder + '/%03i-' % job + socket.gethostname() + '-%i' % self.device
        #caffe.Net.init_glog(log);

        solver = caffe_pb2.SolverParameter()
        net = caffe_pb2.NetParameter()
        test2 = caffe_pb2.NetParameter()
        text_format.Merge(open('examples/cfw_v3/solver.prototxt').read(), solver)
        text_format.Merge(caffe.Net.load_imports('examples/cfw_v3/train_test.prototxt'), net)
        text_format.Merge(caffe.Net.load_imports('examples/cfw_v3/lfw_test.prototxt'), test2)

        solver.device_id = self.device
        solver.stepsize = stepsize
        solver.base_lr = base_lr
        solver.snapshot = 0
        solver.snapshot_prefix = log
        changes = 0

        for l in net.layers:
            if 'conv1' in l.name:
                l.convolution_param.num_output = int(output0)
                changes += 1;
            if 'conv2' in l.name:
                l.convolution_param.num_output = int(output1)
                changes += 1;
            if 'conv3' in l.name:
                l.convolution_param.num_output = int(output2)
                changes += 1;

            if 'ip1' in l.name:
                l.inner_product_param.num_output = int(btlneck)
                changes += 1;

            if l.name == 'a_loss/softmax_loss':
                l.loss_weight[0] = loss_weights[0]
                changes += 1;
            if l.name == 'b_loss/softmax_loss':
                l.loss_weight[0] = loss_weights[1]
                changes += 1;
            if l.name == 'contrast/siamese_hinge_loss':
                l.loss_weight[0] = loss_weights[2]
                changes += 1;

        for l in test2.layers:
            if 'conv1' in l.name:
                l.convolution_param.num_output = int(output0)
                changes += 1;
            if 'conv2' in l.name:
                l.convolution_param.num_output = int(output1)
                changes += 1;
            if 'conv3' in l.name:
                l.convolution_param.num_output = int(output2)
                changes += 1;

            if 'ip1' in l.name:
                l.inner_product_param.num_output = int(btlneck)
                changes += 1;

        if changes != 19:
            raise

        _, net_file = mkstemp()
        _, ts2_file = mkstemp()
        text_format.PrintMessage(net, open(net_file, 'w'))
        text_format.PrintMessage(test2, open(ts2_file, 'w'))
        solver.net = net_file
        solver.test_net[0] = ts2_file
        
        _, solver_file = mkstemp()
        text_format.PrintMessage(solver, open(solver_file, 'w'))

        cmd = ""
        cmd += "GLOG_logtostderr=0 GLOG_log_dir=%s/" % log
        cmd += " ./build/tools/caffe train "
        cmd += " --solver=%s" % solver_file
        cmd += " --gpu=%s" % self.device

        print "cmd", cmd
        print "folder", self.folder
        print "log", log
        os.makedirs(log)
        os.system(cmd)

        #import IPython ; IPython.embed()

#        solver = caffe.SGDSolver(solver_file)
#        solver.net.set_mode_gpu()
#        solver.solve()

        score = 0

        with open("%s/caffe.INFO" % log) as fh:
            score = parse_score(fh)
        return 1 - score

if __name__ == '__main__':
    main()
