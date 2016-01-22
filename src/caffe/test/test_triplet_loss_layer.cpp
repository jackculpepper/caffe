#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(6, 1, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(6, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~TripletLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // manually compute to compare
  const Dtype margin = layer_param.triplet_loss_param().margin();
  const int num_triplets = this->blob_bottom_data_->num()/3;
  const int channels = this->blob_bottom_data_->channels();

  Dtype loss(0);

  for (int i = 0; i < num_triplets; ++i) {
    Dtype dist_sq_pos(0);
    Dtype dist_sq_neg(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff_pos = this->blob_bottom_data_->cpu_data()[3*i*channels+j] -
        this->blob_bottom_data_->cpu_data()[(3*i+1)*channels+j];
      dist_sq_pos += diff_pos*diff_pos;

      Dtype diff_neg = this->blob_bottom_data_->cpu_data()[3*i*channels+j] -
        this->blob_bottom_data_->cpu_data()[(3*i+2)*channels+j];
      dist_sq_neg += diff_neg*diff_neg;

    }
    loss += std::max(margin + dist_sq_pos - dist_sq_neg, Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(num_triplets) / Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(TripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
