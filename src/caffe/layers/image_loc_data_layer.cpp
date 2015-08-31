#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>


#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"


using std::iterator;
using std::string;
using std::pair;

using namespace cv;

namespace caffe {

template <typename Dtype>
ImageLocDataLayer<Dtype>::~ImageLocDataLayer<Dtype>() {
  this->StopInternalThread();
}


bool LocReadImageToDatum(const vector<string>& files, const int height,
		const int width, Datum* datum)
{

	// files:
	// image, normals * n, labels

	cv::Mat cv_img;
	string filename;
	if (height > 0 && width > 0)
	{
		filename = files[0];
		cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	}
	else
	{
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}

	if (!cv_img.data)
	{
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}

	// the last file is label
	int filenum = files.size() - 1;

	datum->set_channels(3 * filenum);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();
	datum->clear_data();
	datum->clear_float_data();

	for (int c = 0; c < 3; ++c)
	{
		for (int h = 0; h < cv_img.rows; ++h)
		{
			for (int w = 0; w < cv_img.cols; ++w)
			{
				if (filenum == 1)
				{
					datum->add_float_data(
							static_cast<uint8_t>(cv_img.at<cv::Vec3b>(h, w)[c]));
				}
				else
				{
					datum->add_float_data(
							static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
				}

			}
		}
	}

	for (int i = 1; i < filenum; i++)
	{
		string filename2 = files[i];
		//FILE *pFile = fopen(filename2.c_str(), "rb");
		FILE *pFile = fopen(filename2.c_str(), "r");
		for (int c = 0; c < 3; ++c)
		{
			for (int h = 0; h < cv_img.rows; ++h)
			{
				for (int w = 0; w < cv_img.cols; ++w)
				{
					float tnum;
					//fread(&tnum, sizeof(float), 1, pFile);
					fscanf(pFile, "%f", &tnum);
					//tnum = tnum * 128 + 128;
					datum->add_float_data(tnum);
				}
			}
		}
		fclose(pFile);
	}

	int num, cnt = 0;
	string labelfile = files[filenum];
	FILE *fid = fopen(labelfile.c_str(), "r");
	while (fscanf(fid, "%d", &num) > 0)
	{
		datum->add_label(num);
		cnt++;
	}
	fclose(fid);

	return true;
}


template <typename Dtype>
void ImageLocDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_loc_data_param().new_height();
  const int new_width  = this->layer_param_.image_loc_data_param().new_width();
  //const bool is_color  = this->layer_param_.image_loc_data_param().is_color();
  //string root_folder = this->layer_param_.image_loc_data_param().root_folder();
  const int source_num = this->layer_param_.image_loc_data_param().source_num();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_loc_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
    while (infile >> filename)
	{
		vector<string> ts;
		ts.push_back(filename);
		for (int i = 0; i < source_num; i++)
		{
			infile >> filename;
			ts.push_back(filename);
		}
		lines_.push_back(ts);

	}

  if (this->layer_param_.image_loc_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_loc_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_loc_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }


  Datum datum;
  CHECK( LocReadImageToDatum(lines_[lines_id_], new_height, new_width, &datum));
  const int crop_size = this->layer_param_.image_loc_data_param().crop_size();
  const int batch_size = this->layer_param_.image_loc_data_param().batch_size();
  const string& mean_file = this->layer_param_.image_loc_data_param().mean_file();
  vector<int> top_shape;
  if (crop_size > 0)
  {
	  top_shape.push_back(1); top_shape.push_back(datum.channels());
	  top_shape.push_back(crop_size); top_shape.push_back(crop_size);
  }
  else
  {
	  LOG(ERROR) << "should crop" ;
	  //top_shape.push_back(1); top_shape.push_back(datum.channels());
	  //top_shape.push_back(datum.height()); top_shape.push_back(datum.width());
  }
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
  }

  if (this->layer_param_.image_loc_data_param().has_mean_file())
  {
	BlobProto blob_proto;
	LOG(INFO) << "Loading mean file from" << mean_file;
	ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
	data_mean_.FromProto(blob_proto);
	CHECK_EQ(data_mean_.num(), 1);
	CHECK_EQ(data_mean_.channels(), datum.channels());
	CHECK_EQ(data_mean_.height(), crop_size);
	CHECK_EQ(data_mean_.width(), crop_size);
  }
  else
  {
	  LOG(ERROR) << "should have mean" ;
  }
  data_mean_.cpu_data();

}

template <typename Dtype>
void ImageLocDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageLocDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageLocDataParameter image_loc_data_param = this->layer_param_.image_loc_data_param();
  const int batch_size = image_loc_data_param.batch_size();
  const int new_height = image_loc_data_param.new_height();
  const int new_width = image_loc_data_param.new_width();
  //string root_folder = image_loc_data_param.root_folder();
  const int crop_size = this->layer_param_.image_loc_data_param().crop_size();
  const int lines_size = lines_.size();

  const Dtype* mean = data_mean_.cpu_data();
  Datum datum;
  if (!LocReadImageToDatum(lines_[lines_id_], new_height, new_width, &datum))
  {
  		string filename = lines_[lines_id_][0];
  		LOG(ERROR) << "Could not fretch file " << filename;
  		return reinterpret_cast<void*>(NULL);
  }

	vector<int> top_shape;
	if (crop_size > 0)
	{
	  top_shape.push_back(1); top_shape.push_back(datum.channels());
	  top_shape.push_back(crop_size); top_shape.push_back(crop_size);
	}
	else
	{
	  LOG(ERROR) << "should crop" ;
	  //top_shape.push_back(1); top_shape.push_back(datum.channels());
	  //top_shape.push_back(datum.height()); top_shape.push_back(datum.width());
	}
	this->transformed_data_.Reshape(top_shape);
	top_shape[0] = batch_size;
	batch->data_.Reshape(top_shape);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();

	const int slide_stride = image_loc_data_param.slide_stride();
	int hnum = new_height / slide_stride;
	int wnum = new_width / slide_stride;
	int height = new_height;
	int width  = new_width;
	int channels = datum.channels();
	CHECK_EQ(hnum * wnum, batch_size);
	int item_id = 0;
	for (int gh = 0; gh < hnum; gh++)
	for (int gw = 0; gw < wnum; gw++)
	{
		int midh = gh * slide_stride + slide_stride / 2;
		int midw = gw * slide_stride + slide_stride / 2;
		int h_off = midh - crop_size / 2;
		int w_off = midw - crop_size / 2;

		// Normal copy
		for (int c = 0; c < channels; ++c)
		{
			for (int h = 0; h < crop_size; ++h)
			{
				for (int w = 0; w < crop_size; ++w)
				{
					int top_index = ((item_id * channels + c) * crop_size
							+ h) * crop_size + w;
					int data_index = (c * height + h + h_off) * width + w
							+ w_off;
					int mean_index = (c * crop_size + h) * crop_size + w;

					Dtype datum_element = 0;
					if (h + h_off >= 0 && h + h_off < height
							&& w + w_off >= 0 && w + w_off < width)
						datum_element = datum.float_data(data_index);
					prefetch_data[top_index] = ((datum_element)
							- mean[mean_index]);


				}
			}
		}
		CHECK(datum.label_size() == 1);
		for (int label_i = 0; label_i < datum.label_size(); label_i++)
		{
			prefetch_label[item_id * datum.label_size() + label_i] = datum.label(
					label_i);
		}
		item_id++;

	}

	lines_id_++;
	if (lines_id_ >= lines_size) {
	  // We have reached the end. Restart from the first.
	  DLOG(INFO) << "Restarting data prefetching from start.";
	  lines_id_ = 0;
	  if (this->layer_param_.image_loc_data_param().shuffle()) {
		ShuffleImages();
	  }
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

INSTANTIATE_CLASS(ImageLocDataLayer);
REGISTER_LAYER_CLASS(ImageLocData);

}  // namespace caffe
