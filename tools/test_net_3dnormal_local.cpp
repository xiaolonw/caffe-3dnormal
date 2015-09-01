
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;


#define HEIGHT 195
#define WIDTH  260
#define STRIDE 13
#define FILENUM 1

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

void parsename(char* filename, string &res, string &res2)
{
	int len = strlen(filename);
	int id = 0;
	for (int i = len - 1; i >= 0; i --)
	{
		if (filename[i] == '/')
		{
			id = i + 1;
			break;
		}
	}
	res = "";
	for (int i = id; i < len - 4; i ++)
		res = res + filename[i];
	res2 = res + "_edge.txt";
	res = res + ".txt";
}


string numtostr(int num)
{
	string res = "";
	while (num > 0)
	{
		res = res + (char)(num % 10 + '0');
		num = num / 10;
	}
	int len =res.size();
	for(int i = 0; i < len / 2; i ++)
	{
		char temp = res[i];
		res[i] = res[len - 1 - i];
		res[len - 1 - i] = temp;
	}
	return res;
}

char buf[101000];
int main(int argc, char** argv)
{

	//Caffe::set_phase(caffe::TEST);
	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}

	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(3);

	/*
	std::string test_net_param_s(argv[1]);
	Net<float> caffe_test_net(test_net_param_s, caffe::TEST);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
	*/

	std::string test_net_param_s(argv[1]);
	Net<float> caffe_test_net(test_net_param_s, caffe::TEST);
	std::string load_net_param_s(argv[2]);
	caffe_test_net.CopyTrainedLayersFrom(load_net_param_s);


	string labelFile(argv[3]);
	int data_counts = 0;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fgets(buf,100000,file) > 0)
	{
		data_counts++;
	}
	fclose(file);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	float output;
	int counts = 0;

	file = fopen(labelFile.c_str(), "r");
	int batchCount = data_counts;

	float * output_mat = new float[HEIGHT * WIDTH * 3];
	float * output_edge = new float[HEIGHT * WIDTH * 3];

	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
		Blob<float>* edges = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
		int bsize = bboxs->num();

		const Blob<float>* labels = (*(caffe_test_net.bottom_vecs().rbegin()))[1];

		char fname[1010];
		char fname2[1010];
		fscanf(file, "%s", fname);
		for(int i = 0; i < FILENUM; i ++ ) fscanf(file, "%s", fname2);

		string filename, filename2;
		parsename(fname, filename, filename2);
		filename = rootfolder + "/" + filename;
		filename2= rootfolder + "/" + filename2;
		printf("%s\n", filename.c_str());
		FILE * resultfile = fopen(filename.c_str(), "w");
		FILE * resultfile2 = fopen(filename2.c_str(), "w");


		//fprintf(resultfile, "%s ", fname);

		int channels = bboxs->channels();
		int height   = bboxs->height();
		int width    = bboxs->width();
		int hnum = HEIGHT / STRIDE;
		int wnum = WIDTH / STRIDE;
		int stride = STRIDE;
		for(int i = 0; i < bsize; i++)
		for(int c = 0; c < channels; c ++)
		{
			int hi = i / wnum;
			int wi = i % wnum;
			int off_h = hi * stride;
			int off_w = wi * stride;
			for(int h = 0; h < height; h ++)
				for(int w = 0; w < width; w ++)
					{
						//output_mat[c * HEIGHT * WIDTH + (off_w + w) * HEIGHT + off_h + h ] = (float)(bboxs->data_at(i, c, h, w));
						output_mat[c * HEIGHT * WIDTH + (off_h + h) * WIDTH + off_w + w ] = (float)(bboxs->data_at(i, c, h, w)) * 128 + 128;
					}
		}
		for(int i = 0; i < HEIGHT * WIDTH * 3; i ++)
			fprintf(resultfile, "%f ", output_mat[i] );


		int channels2 = edges->channels();

		for(int i = 0; i < bsize; i++)
		{
			int hi = i / wnum;
			int wi = i % wnum;
			int off_h = hi * stride;
			int off_w = wi * stride;

			vector<float> props;
			for(int j = 0; j < 3; j ++) props.push_back(0);

			for(int c = 1; c < channels2; c ++)
			{
				int colorid = (c - 1) / 12;
				props[colorid] = props[colorid] + (float)(edges->data_at(i, c, 0, 0));
			}

			for(int c = 0; c < 3; c ++)
			for(int h = 0; h < height; h ++)
				for(int w = 0; w < width; w ++)
				{
					output_edge[c * HEIGHT * WIDTH + (off_h + h) * WIDTH + off_w + w ] = props[c] * 255;
				}

		}
		for(int i = 0; i < HEIGHT * WIDTH * 3; i ++)
			fprintf(resultfile2, "%f ", output_edge[i] );



		fclose(resultfile);
		fclose(resultfile2);



	}

	delete output_mat;
	delete output_edge;

	//fclose(resultfile);
	fclose(file);

	return 0;
}
