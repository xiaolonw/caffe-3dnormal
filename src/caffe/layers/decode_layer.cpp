// Copyright 2014 BVLC and contributors.

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void DecodeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "IP Layer takes a single blob as output.";

  int out_channel = 3;
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  const string& source_dict = this->layer_param_.decode_param().source_dict();
  const string& source_triIDs = this->layer_param_.decode_param().source_triids();


  int temp_i;
  float temp_f;
  dict.clear();
  FILE * fid = fopen(source_dict.c_str(), "r");
  while(fscanf(fid, "%f", &temp_f) > 0 )
  {
	  vector<float> temp;
	  temp.push_back(temp_f);
	  for(int i = 0; i < 2; i ++)
      {
		  fscanf(fid, "%f", &temp_f);
		  temp.push_back(temp_f);
	  }
	  dict.push_back(temp);
  }
  fclose(fid);

  fid = fopen(source_triIDs.c_str(), "r");
  while(fscanf(fid, "%d", &temp_i) > 0 )
  {
	  vector<int> temp;
	  temp.push_back(temp_i);
	  for(int i = 0; i < 2; i ++)
      {
		  fscanf(fid, "%d", &temp_i);
		  temp.push_back(temp_i);
	  }
	  triIDs.push_back(temp);
  }
  fclose(fid);
  CHECK_EQ(dict.size(), channel);


}


template <typename Dtype>
void DecodeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top)
{
	int out_channel = 3;
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	top[0]->Reshape(num, out_channel, height, width);
}




template <typename Dtype>
void DecodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int dimout = top[0]->count() / bottom[0]->num();
    int channel = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    int imgSize = height * width;

    CHECK_EQ(dimout, imgSize * 3);
    for(int i = 0; i < top[0]->count(); i ++)
    {
    	top_data[i] = 0;
    }

    bool usesoft = this->layer_param_.decode_param().usesoft();

    if (usesoft)
    {
    	for (int i = 0; i < num; i ++)
		{
			for(int h = 0; h < height; h ++)
			for(int w = 0; w < width; w ++)
			{
				float xv = 0,yv = 0,zv = 0;
				for(int c = 0; c < channel; c ++)
				{
					int id = i * dim + c * imgSize + h * width + w;
					float pt = bottom_data[id];
					xv += pt * dict[c][0];
					yv += pt * dict[c][1];
					zv += pt * dict[c][2];
				}

				top_data[i * dimout  + h * width + w] = xv;
				top_data[i * dimout  + 1 * imgSize + h * width + w] = yv;
				top_data[i * dimout  + 2 * imgSize + h * width + w] = zv;
			}
		}
    }
    else
    {
    	for (int i = 0; i < num; i ++)
		{
			for(int h = 0; h < height; h ++)
			for(int w = 0; w < width; w ++)
			{
				float maxp = 0;
				int maxtr = 0;
				for(int tr = 0; tr < triIDs.size(); tr ++)
				{
					float pt = 0;
					for(int ti = 0; ti < triIDs[tr].size(); ti ++)
					{
						int c = triIDs[tr][ti] - 1;
						int id = i * dim + c * imgSize + h * width + w;
						pt += bottom_data[id];
					}
					if (pt > maxp)
					{
						maxp = pt;
						maxtr = tr;
					}
				}
				float xv = 0,yv = 0,zv = 0;
				for(int ti = 0; ti < triIDs[maxtr].size(); ti ++)
				{
					int c = triIDs[maxtr][ti] - 1;
					int id = i * dim + c * imgSize + h * width + w;
					float pt = bottom_data[id];
					xv += pt * dict[c][0];
					yv += pt * dict[c][1];
					zv += pt * dict[c][2];
				}

				top_data[i * dimout  + h * width + w] = xv;
				top_data[i * dimout  + 1 * imgSize + h * width + w] = yv;
				top_data[i * dimout  + 2 * imgSize + h * width + w] = zv;
			}
		}
    }

}

template <typename Dtype>
void DecodeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	int num = bottom[0]->num();
	int dim = bottom[0]->count()/ bottom[0]->num();
	int dimout = top[0]->count() / top[0]->num();
	int channel = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int imgSize = height * width;
	int out_channel = top[0]->channels();

	for(int i = 0; i < bottom[0]->count(); i ++)
		bottom_diff[i] = 0;

	for (int i = 0; i < num; i ++)
	{
		for(int h = 0; h < height; h ++)
		for(int w = 0; w < width; w ++)
		{
			float maxp = 0;
			int maxtr = 0;
			for(int tr = 0; tr < triIDs.size(); tr ++)
			{
				float pt = 0;
				for(int ti = 0; ti < triIDs[tr].size(); ti ++)
				{
					int c = triIDs[maxtr][ti] - 1;
					int id = i * dim + c * imgSize + h * width + w;
					pt += bottom_data[id];
				}
				if (pt > maxp)
				{
					maxp = pt;
					maxtr = tr;
				}
			}
			float xv = 0,yv = 0,zv = 0;
			for(int ti = 0; ti < triIDs[maxtr].size(); ti ++)
			{
				int c = triIDs[maxtr][ti] - 1;
				int id = i * dim + c * imgSize + h * width + w;
				for(int k = 0; k < out_channel; k ++)
				{
					int id2 = i * dimout  + k * imgSize + h * width + w;
					bottom_diff[id] += dict[c][k] * top_diff[id2];
				}
			}

		}
	}
}

INSTANTIATE_CLASS(DecodeLayer);
REGISTER_LAYER_CLASS(Decode);



}  // namespace caffe
