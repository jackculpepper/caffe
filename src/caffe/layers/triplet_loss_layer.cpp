#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // dimension of each descriptor
  int dim = bottom[0]->count()/bottom[0]->num();

  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  int num_triplets = bottom[0]->num()/3;

  diff_pos.Reshape(num_triplets, dim, 1, 1);
  diff_neg.Reshape(num_triplets, dim, 1, 1);
  loss_i_.Reshape(num_triplets, 1, 1, 1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();

  // batch size must be a multiple of 3
  CHECK_EQ(bottom[0]->num() % 3, 0);

  Dtype dist_sq_pos(0.0);
  Dtype dist_sq_neg(0.0);
  Dtype loss(0.0);

  int dim = bottom[0]->count()/bottom[0]->num();
  int num_triplets = bottom[0]->num()/3;

  for (int i = 0; i < num_triplets; ++i) {
    caffe_sub(dim,
              bottom[0]->cpu_data() + 3*i*dim,       // x_a
              bottom[0]->cpu_data() + (3*i + 1)*dim, // x_p
              diff_pos.mutable_cpu_data() + i*dim);  // x_a - x_p
    dist_sq_pos = 
      caffe_cpu_dot(dim,
                    diff_pos.cpu_data() + i*dim,
                    diff_pos.cpu_data() + i*dim);

    caffe_sub(dim,
              bottom[0]->cpu_data() + 3*i*dim,       // x_a
              bottom[0]->cpu_data() + (3*i + 2)*dim, // x_n
              diff_neg.mutable_cpu_data() + i*dim);  // x_a - x_n
    dist_sq_neg = 
      caffe_cpu_dot(dim,
                    diff_neg.cpu_data() + i*dim,
                    diff_neg.cpu_data() + i*dim);

    loss_i_.mutable_cpu_data()[i] =
      std::max(margin + dist_sq_pos - dist_sq_neg, Dtype(0.0));

    loss += loss_i_.mutable_cpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(num_triplets) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  int dim = bottom[0]->count()/bottom[0]->num();
  int num_triplets = bottom[0]->num()/3;

  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] /
      static_cast<Dtype>(num_triplets);

    for (int i = 0; i < num_triplets; ++i) {
      Dtype* bout = bottom[0]->mutable_cpu_diff();

      if (loss_i_.mutable_cpu_data()[i] > Dtype(0.0)) {
        // contribution to dE/dx_a from x_a - x_p term
        caffe_cpu_axpby(dim,
                        alpha,
                        diff_pos.cpu_data() + i*dim,
                        Dtype(1.0),
                        bout + 3*i*dim);

        // contribution to dE/dx_a from x_a - x_n term
        caffe_cpu_axpby(dim,
                        -alpha,
                        diff_neg.cpu_data() + i*dim,
                        Dtype(1.0),
                        bout + 3*i*dim);

        // contribution to dE/dx_p from x_a - x_p term
        caffe_cpu_axpby(dim,
                        -alpha,
                        diff_pos.cpu_data() + i*dim,
                        Dtype(1.0),
                        bout + (3*i + 1)*dim);

        // contribution to dE/dx_n from x_a - x_n term
        caffe_cpu_axpby(dim,
                        alpha,
                        diff_neg.cpu_data() + i*dim,
                        Dtype(1.0),
                        bout + (3*i + 2)*dim);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe


