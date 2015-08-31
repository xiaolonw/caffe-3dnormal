// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

// bottom: channel * h * w inputs, channel = class numbers

namespace caffe
{

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{

	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
	sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
			bottom[0]->width());
	Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
	for (int i = 0; i < sum_multiplier_.count(); ++i)
	{
		multiplier_data[i] = 1.;
	}
	//wxl
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	int dimClass = bottom[0]->channels();
	int imgSize = dim / dimClass;
	scale_.Reshape(bottom[0]->num() * imgSize, 1, 1, 1);
}

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* scale_data = scale_.mutable_cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	int dimClass = bottom[0]->channels();
	int height = bottom[0]->height();
	int width  = bottom[0]->width();
	int imgSize = height * width;
	memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());


    // num, channel, height, width

	// we need to subtract the max to avoid numerical issues, compute the exp,
	// and then normalize.
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			scale_data[i * imgSize + j] = bottom_data[i * dim + j];
			for(int k = 0; k < dimClass; ++k)
			{
				scale_data[i * imgSize + j] = max(scale_data[i * imgSize + j], bottom_data[i * dim + k * imgSize + j]);
			}
		}
	}

	// subtraction
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			for(int k = 0; k < dimClass; ++k)
				top_data[i * dim + k * imgSize + j] -= scale_data[i * imgSize + j];
		}
	}
	// Perform exponentiation
	caffe_exp<Dtype>(num * dim, top_data, top_data);

	//sum after exp
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
            scale_data[i * imgSize + j] = 0;
			for(int k = 0; k < dimClass; ++k)
				scale_data[i * imgSize + j] += top_data[i * dim + k * imgSize + j];
		}
	}
	// Do division
	for(int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			for(int k = 0; k < dimClass; ++k)
			{
				top_data[i * dim + k * imgSize + j] /= Dtype(1.) * scale_data[i * imgSize + j];
			}
		}
	}

}

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* scale_data = scale_.mutable_cpu_data();
	int num = top[0]->num();
	int dim = top[0]->count() / top[0]->num();
	int dimClass = bottom[0]->channels();
	int height = bottom[0]->height();
	int width  = bottom[0]->width();
	int imgSize = height * width;
	memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());
	// Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < imgSize; ++j)
		{
			scale_data[i * imgSize + j] = 0;
			for(int k = 0; k < dimClass; ++k)
			{
				scale_data[i * imgSize + j] += top_diff[i * dim + k * imgSize + j] * top_data[i * dim + k * imgSize + j];
			}
		}
	}
	// subtraction
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < imgSize; ++j)
		{
			for(int k = 0; k < dimClass; ++k)
			{
				bottom_diff[i * dim + k * imgSize + j] -= scale_data[i * imgSize + j];
			}
		}
	}
	// elementwise multiplication
	caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);

}

INSTANTIATE_CLASS(MultiSoftmaxLayer);

}  // namespace caffe
