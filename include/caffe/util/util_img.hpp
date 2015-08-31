// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_IMG_H_
#define CAFFE_UTIL_IMG_H_

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width);

template <typename Dtype>
void BiLinearResizeMat_gpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width);

template <typename Dtype>
void GetBiLinearResizeMatRules_cpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);

template <typename Dtype>
void GetBiLinearResizeMatRules_gpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);


}


#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
