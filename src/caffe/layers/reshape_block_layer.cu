#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      Forward_cpu(bottom, top);
}

template <typename Dtype>
void ReshapeBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      Backward_cpu(top, propagate_down, bottom);     
}

INSTANTIATE_LAYER_GPU_FUNCS(ReshapeBlockLayer);

}  // namespace caffe
