// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe
{

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	CHECK_EQ(bottom.size(), 2) << "Multi SoftmaxLoss Layer takes two blobs as input.";
	CHECK_EQ(top.size(), 0) << "Multi SoftmaxLoss Layer takes no blob as output.";
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}



template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	const Dtype* prob_data = prob_.cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	int height = bottom[0]->height();
	int width  = bottom[0]->width();
	int imgSize = height * width;
	Dtype loss = 0;
	for (int i = 0; i < num; ++i)
	{
		for(int j = 0; j < imgSize; j ++)
		{
			int nowlabel = static_cast<int>(label[i * imgSize + j]); //- 1;
			loss += -log(max( prob_data[i * dim + nowlabel * imgSize + j], Dtype(FLT_MIN)));
		}

	}
	top[0]->mutable_cpu_data()[0] = loss / num;
}

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	// Compute the diff
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* prob_data = prob_.cpu_data();
	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	const Dtype* label = bottom[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	//int dimClass = bottom[0]->channels();
	int height = bottom[0]->height();
	int width  = bottom[0]->width();
	int imgSize = height * width;

	for (int i = 0; i < num; ++i)
	{
		for(int j = 0; j < imgSize; j ++)
		{
			int nowlabel = static_cast<int>(label[i * imgSize + j]) ;//- 1;
			bottom_diff[ i * dim + nowlabel * imgSize + j  ] -= 1;
		}
	}
	// Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
	// LossParameter loss_param = this->layer_param_.loss_param();
	// caffe_scal(prob_.count(), Dtype(loss_param.scale()), bottom_diff);
	caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(MultiSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultiSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultiSoftmaxWithLoss);

}  // namespace caffe
