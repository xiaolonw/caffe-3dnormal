// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";

  const int out_channel = this->layer_param_.reshape_block_param().new_channel();
  const int out_height = this->layer_param_.reshape_block_param().new_height();
  const int out_width = this->layer_param_.reshape_block_param().new_width();
  const int patch_width = this->layer_param_.reshape_block_param().new_patch_width();
  const int patch_height = this->layer_param_.reshape_block_param().new_patch_height();

  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  CHECK_EQ(patch_width * patch_height * out_channel, channel);
  CHECK_EQ(channel * height * width, out_channel * out_height * out_width);

}

template <typename Dtype>
void ReshapeBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top)
{
	const int out_channel = this->layer_param_.reshape_block_param().new_channel();
	const int out_height = this->layer_param_.reshape_block_param().new_height();
	const int out_width = this->layer_param_.reshape_block_param().new_width();
	int num = bottom[0]->num();

	top[0]->Reshape(num, out_channel, out_height, out_width);
}


template <typename Dtype>
void ReshapeBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->mutable_cpu_data();

    const int out_channel = this->layer_param_.reshape_block_param().new_channel();
    const int out_height = this->layer_param_.reshape_block_param().new_height();
    const int out_width = this->layer_param_.reshape_block_param().new_width();
    const int patch_width = this->layer_param_.reshape_block_param().new_patch_width();
    const int patch_height = this->layer_param_.reshape_block_param().new_patch_height();

    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int dim = height * width * channel;

    for(int i = 0; i < top[0]->count(); i ++)
	{
		top_data[i] = 0;
	}

    for (int i = 0; i < num; i ++)
    {
    	Dtype* top_data_now = top_data + dim * i;
    	const Dtype* bottom_data_now = bottom_data + dim * i;
		for(int h = 0; h < height; h ++) for(int w = 0; w < width; w ++)
		{
			int offw = w * patch_width;
			int offh = h * patch_height;
			int cnt  = 0;
			// first w then h, this is because matlab
			for (int nw = 0; nw < patch_width; nw ++)
			for(int nh = 0; nh < patch_height; nh ++)
			for(int nc = 0; nc < out_channel; nc ++)
			{
				int nowh  = offh + nh, noww = offw + nw;
				int idout = nc * out_height * out_width + nowh * out_width + noww;
				int idin  = cnt; //nw * patch_height * out_channel + nh * out_channel + nc;
				int idin2 = cnt * height * width + h * width + w;

				top_data_now[idout] = bottom_data_now[idin2];

				cnt ++;
			}
		}
    }

}

template <typename Dtype>
void ReshapeBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	const int out_channel = this->layer_param_.reshape_block_param().new_channel();
	const int out_height = this->layer_param_.reshape_block_param().new_height();
	const int out_width = this->layer_param_.reshape_block_param().new_width();
	const int patch_width = this->layer_param_.reshape_block_param().new_patch_width();
	const int patch_height = this->layer_param_.reshape_block_param().new_patch_height();

	int num = bottom[0]->num();
	int channel = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int dim = height * width * channel;

	for(int i = 0; i < bottom[0]->count(); i ++)
			bottom_diff[i] = 0;


	for (int i = 0; i < num; i ++)
	{
		Dtype* bottom_diff_now = bottom_diff + dim * i;
		const Dtype* top_diff_now = top_diff + dim * i;
		for(int h = 0; h < height; h ++) for(int w = 0; w < width; w ++)
		{
			int offw = w * patch_width;
			int offh = h * patch_height;
			int cnt  = 0;
			for (int nw = 0; nw < patch_width; nw ++)
			for(int nh = 0; nh < patch_height; nh ++)
			for(int nc = 0; nc < out_channel; nc ++)
			{
				int nowh  = offh + nh, noww = offw + nw;
				int idout = nc * out_height * out_width + nowh * out_width + noww;
				int idin  = cnt; //nw * patch_height * out_channel + nh * out_channel + nc;
				int idin2 = cnt * height * width + h * width + w;

				bottom_diff_now[idin2] = top_diff_now[idout];

				cnt ++;
			}
		}
	}
}

INSTANTIATE_CLASS(ReshapeBlockLayer);
REGISTER_LAYER_CLASS(ReshapeBlock);

}  // namespace caffe
