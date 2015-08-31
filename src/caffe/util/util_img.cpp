
/**
 * contributed by Zhujin Liang and Lichao Huang
 */
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"


namespace caffe {


template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	int loop_n = dst_height * dst_width;
	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;

		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name() )
		{
			src_h = floor(fh);
		}
		else
		{
			src_h = floorf(fh);
		}

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		int src_w ;//= floor(fw);
		if(typeid(Dtype).name() == typeid(double).name() )
		{
			src_w = floor(fw);
		}
		else
		{
			src_w = floorf(fw);
		}
		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		dst_data[dst_idx] += (w_h0 * w_w0 * src_data[src_idx]);
		if (src_w + 1 < src_width)
			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		if (src_h + 1 < src_height)
			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);

		if (src_w + 1 < src_width && src_h + 1 < src_height)
			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
	}

}


template void BiLinearResizeMat_cpu(const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width);

template void BiLinearResizeMat_cpu(const double* src, const int src_height, const int src_width,
		double* dst, const int dst_height, const int dst_width);


template <typename Dtype>
void GetBiLinearResizeMatRules_cpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;
	caffe::caffe_set(loop_n,(Dtype)0,loc1);
	caffe::caffe_set(loop_n,(Dtype)0,loc2);
	caffe::caffe_set(loop_n,(Dtype)0,loc4);
	caffe::caffe_set(loop_n,(Dtype)0,loc3);

	caffe::caffe_set(loop_n,(Dtype)0,weight1);
	caffe::caffe_set(loop_n,(Dtype)0,weight2);
	caffe::caffe_set(loop_n,(Dtype)0,weight3);
	caffe::caffe_set(loop_n,(Dtype)0,weight4);

	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name())
			 src_h = floor(fh);
		else
			 src_h = floorf(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;

		int src_w ;
		if(typeid(Dtype).name() == typeid(double).name())
			src_w = floor(fw);
		else
			src_w = floorf(fw);

		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
//		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = static_cast<Dtype>(src_idx);
		weight1[dst_idx] = w_h0 * w_w0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = static_cast<Dtype>(src_idx + 1);
			weight2[dst_idx] = w_h0 * w_w1;
//			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
//			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = static_cast<Dtype>(src_idx + src_width);
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = static_cast<Dtype>(src_idx + src_width + 1);
			weight4[dst_idx] = w_h1 * w_w1;
//			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}

	}

}


template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		float* loc1, float* weight1, float* loc2, float* weight2,
				float* loc3, float* weight3, float* loc4, float* weight4);

template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		double* loc1, double* weight1, double* loc2, double* weight2,
				double* loc3, double* weight3, double* loc4, double* weight4);




template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c) {


	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;


	const Dtype* src_data = &(src->cpu_data()[src_offset]);
	Dtype* dst_data = &(dst->mutable_cpu_data()[dst_offset]);
	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
}

template void ResizeBlob_cpu(const Blob<float>* src, const int src_n, const int src_c,
		Blob<float>* dst, const int dst_n, const int dst_c);
template void ResizeBlob_cpu(const Blob<double>* src, const int src_n, const int src_c,
		Blob<double>* dst, const int dst_n, const int dst_c);


template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst);



template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4){

	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	GetBiLinearResizeMatRules_cpu(  src->height(),src->width(),
			 dst->height(), dst->width(),
			loc1->mutable_cpu_data(), loc1->mutable_cpu_diff(), loc2->mutable_cpu_data(), loc2->mutable_cpu_diff(),
			loc3->mutable_cpu_data(), loc3->mutable_cpu_diff(), loc4->mutable_cpu_data(), loc4->mutable_cpu_diff());

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst,
		Blob<float>* loc1, Blob<float>* loc2, Blob<float>* loc3, Blob<float>* loc4);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst,
		Blob<double>* loc1, Blob<double>* loc2, Blob<double>* loc3, Blob<double>* loc4);

}
// namespace caffe
