#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(
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
    caffe_gpu_sub(dim,
                  bottom[0]->gpu_data() + 3*i*dim,       // x_a
                  bottom[0]->gpu_data() + (3*i + 1)*dim, // x_p
                  diff_pos.mutable_gpu_data() + i*dim);  // x_a - x_p
    caffe_gpu_dot(dim,
                  diff_pos.gpu_data() + i*dim,
                  diff_pos.gpu_data() + i*dim,
                  &dist_sq_pos);

    caffe_gpu_sub(dim,
                  bottom[0]->gpu_data() + 3*i*dim,       // x_a
                  bottom[0]->gpu_data() + (3*i + 2)*dim, // x_n
                  diff_neg.mutable_gpu_data() + i*dim);  // x_a - x_n
    caffe_gpu_dot(dim,
                  diff_neg.gpu_data() + i*dim,
                  diff_neg.gpu_data() + i*dim,
                  &dist_sq_neg);

    loss_i_.mutable_cpu_data()[i] =
      std::max(margin + dist_sq_pos - dist_sq_neg, Dtype(0.0));

    loss += loss_i_.mutable_cpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(num_triplets) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int dim = bottom[0]->count()/bottom[0]->num();
  int num_triplets = bottom[0]->num()/3;

  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] /
      static_cast<Dtype>(num_triplets);

    for (int i = 0; i < num_triplets; ++i) {
      Dtype* bout = bottom[0]->mutable_gpu_diff();

      if (loss_i_.mutable_cpu_data()[i] > Dtype(0.0)) {
        // contribution to dE/dx_a from x_a - x_p term
        caffe_gpu_axpby(dim,
                        alpha,
                        diff_pos.gpu_data() + i*dim,
                        Dtype(1.0),
                        bout + 3*i*dim);

        // contribution to dE/dx_a from x_a - x_n term
        caffe_gpu_axpby(dim,
                        -alpha,
                        diff_neg.gpu_data() + i*dim,
                        Dtype(1.0),
                        bout + 3*i*dim);

        // contribution to dE/dx_p from x_a - x_p term
        caffe_gpu_axpby(dim,
                        -alpha,
                        diff_pos.gpu_data() + i*dim,
                        Dtype(1.0),
                        bout + (3*i + 1)*dim);

        // contribution to dE/dx_n from x_a - x_n term
        caffe_gpu_axpby(dim,
                        alpha,
                        diff_neg.gpu_data() + i*dim,
                        Dtype(1.0),
                        bout + (3*i + 2)*dim);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
