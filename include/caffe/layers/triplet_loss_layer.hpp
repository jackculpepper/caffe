#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
  class TripletLossLayer : public LossLayer<Dtype> {
   public:
    explicit TripletLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline const char* type() const { return "TripletLoss"; }
    /**
     * Like most loss layers, in the TripletLossLayer we can backpropagate
     * to the first input. The second input is a label that is not actually used,
     * because the loss is calculated under the presumption that the batch elements
     * are 3-tuples with this order: anchor, positive, negative.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return bottom_index != 1;
    }

   protected:
    /// @copydoc TripletLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    Blob<Dtype> diff_pos;
    Blob<Dtype> diff_neg;
    Blob<Dtype> loss_i_;
  };

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
