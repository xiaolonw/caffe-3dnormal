# caffe-3dnormal

This code is developed based on Caffe: [project site](http://caffe.berkeleyvision.org).

This code is the implementation for training the siamese-triplet network in the paper:

**Xiaolong Wang**, David F. Fouhey and Abhinav Gupta. Designing Deep Networks for Surface Normal Estimation. Proc. of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [pdf](http://www.cs.cmu.edu/~xiaolonw/papers/deep3d.pdf)

BibTeX: 
```txt
@inproceedings{Wang_SSGAN2016,
    Author = {Xiaolong Wang and David F. Fouhey and Abhinav Gupta},
    Title = {Designing Deep Networks for Surface Normal Estimation},
    Booktitle = {CVPR},
    Year = {2015},
}
```

Instructions
----

This code provides the testing scripts for surface normal estimation. It is developed on relatively older caffe version and can be compiled with CUDA 6.5. 

The results on NYUv2 can be downloaded from here: 
https://www.dropbox.com/sh/k8u7ut3lctg3iyr/AABSggcm6kCSjquTftzTJGT5a?dl=0 
with file name: normals.tar.gz 

The models can be downloaded from the same folder

The scripts for inference are:
caffe-3dnormal/3dscript/global_test/test_3dnet_global.sh (using list: testLabels_single.txt)
caffe-3dnormal/3dscript/local_test/test_3dnet_local.sh (using list: testLabels_loc.txt)

For fusion network, if there is no vanishing point, use:
genlist_fusion2.m generate the list and then run
caffe-3dnormal/3dscript/fusion_test/test_3dnet_local2.sh

If there is vanishing point calculated, use:
genlist_fusion.m generate the list and then run
caffe-3dnormal/3dscript/fusion_test/test_3dnet_local.sh

genimgs.m can be used to visualize the results.

vp.tar.gz are the vanishing points for NYUv2 



