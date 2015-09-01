
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
	res2 = res + "_layout.txt";
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
	string layoutfolder(argv[5]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	float output;
	int counts = 0;

	file = fopen(labelFile.c_str(), "r");

	Blob<float>* c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
    int c2 = c1->num();
	int batchCount = std::ceil(data_counts / (floor)(c2));
	printf("normals: %d %d %d %d\n", c1->num(), c1->channels(), c1->height(), c1->width());

	c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
	printf("layouts: %d %d %d %d\n", c1->num(), c1->channels(), c1->height(), c1->width());

	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
		Blob<float>* layouts = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
		int bsize = bboxs->num();

		for (int i = 0; i < bsize && counts < data_counts; i++, counts++)
		{
			char fname[1010];
			int lbl;
			fscanf(file, "%s", fname);
			fscanf(file, "%d", &lbl);

			string filename, filename2;
			parsename(fname, filename, filename2);
			filename = rootfolder + "/" + filename;
			filename2= rootfolder + "/" + filename2;
			FILE * resultfile = fopen(filename.c_str(), "w");
			//FILE * resultfile2 = fopen(filename2.c_str(), "w");

			int channels = bboxs->channels();
			int height   = bboxs->height();
			int width    = bboxs->width();

			for (int c = 0; c < channels; c ++)
				for(int h = 0; h < height; h ++)
					for(int w = 0; w < width; w ++)
					{
						fprintf(resultfile, "%f ", (float)(bboxs->data_at(i, c, h, w)) * 128 + 128 );
					}
			//fprintf(resultfile, "\n");

			int channels2 = layouts->channels();
			float maxprop = 0;
			int  maxid = 0;
			for (int c = 0; c < channels2; c ++)
			{
				if(maxprop < (float)(layouts->data_at(i, c, 0, 0)) )
				{
					maxprop = (float)(layouts->data_at(i, c, 0, 0)) ;
					maxid = c;
				}
			}

			string strid = numtostr(maxid + 1);
			strid = strid + ".txt";
			strid = layoutfolder + "/" + strid;

			string cmd = "cp " + strid + " " + filename2;
			printf("%s\n", cmd.c_str());

			system(cmd.c_str());




			fclose(resultfile);
		}
	}

	fclose(file);


	return 0;
}
