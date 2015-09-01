#!/usr/bin/env sh                                                                                                

ROOTFILE=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1  /home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint_past/build/tools/test_net_3dnormal_globallayouttxt.bin seg_test_2fc_3dnormal_layout.prototxt /home/dragon123/3dnormal_joint_cnncode/models/global_model/3dnormal__iter_150000  /home/dragon123/3dnormal_joint_cnncode/imgfile_global.txt /home/dragon123/3dnormal_joint_cnncode/slide_results/txts_global /home/dragon123/3dnormal_joint_cnncode/models/global_model/layouttxt



