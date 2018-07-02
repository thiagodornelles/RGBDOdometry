#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "filereading.h"
#include "projector.h"
#include "odometry.h"

using namespace std;
using namespace pcl;
using namespace pcl::visualization;
using namespace boost;
using namespace cv;
using namespace Eigen;

int main (int argc, char **argv){

    Projector projector;
    projector.setIntrisecs(517.3, 516.5, 318.6, 255.3);

    Odometry odometry = Odometry(projector);

    PCLVisualizer viewer("Cloud Viewer");
    viewer.initCameraParameters();
    viewer.setCameraPosition(0,0,0,0,1,0);
    viewer.setBackgroundColor(30/255.f, 70/255.f, 140/255.f);

    //Reading kinect RGB and Depth image folder
    vector<string> filesDepth, filesRGB;
    string imgFolder = "/home/thiago/rgbd_dataset_freiburg1_plant";
    readFilenames(filesDepth, imgFolder + "/depth/");
    readFilenames(filesRGB, imgFolder + "/rgb/");
    uint frame = 0;
    bool paused = true;

    //Pointcloud
    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);

    //Main Loop
    Matrix4f pose = Matrix4f::Identity();    

    while(!viewer.wasStopped () && frame < filesDepth.size() - 1){
        //Reading images
        string depthFilename, rgbFilename;
        depthFilename = imgFolder + "/depth/" + filesDepth[frame];
        rgbFilename = imgFolder + "/rgb/"+ filesRGB[frame];
        Mat refDepth = imread(depthFilename, -1);
        Mat refRGB = imread(rgbFilename);

        depthFilename = imgFolder + "/depth/" + filesDepth[frame+1];
        rgbFilename = imgFolder + "/rgb/"+ filesRGB[frame+1];
        Mat actDepth = imread(depthFilename, -1);
        Mat actRGB = imread(rgbFilename);

        Mat refGray;
        Mat actGray;
        cvtColor(refRGB, refGray, CV_BGR2GRAY);
        cvtColor(actRGB, actGray, CV_BGR2GRAY);

        imshow("Depth", refDepth);
        imshow("RGB", refRGB);
        char key;
        if(!paused){
            if(key == '1'){
                paused = true;
            }
            projector.unprojectDepth(refDepth, refRGB, cloud);
            if(frame == 0){
                viewer.addPointCloud<PointXYZRGB>(cloud);
                viewer.setPointCloudRenderingProperties(PCL_VISUALIZER_POINT_SIZE, 1);
                viewer.resetCamera();
            }
            else{
                pose = pose * odometry.getMatrixRtFromPose6D().inverse();
                transformPointCloud(*cloud, *cloud, pose, true);
                viewer.addPointCloud(cloud, to_string(frame));
                cloud->points.clear();
            }
            frame++;

            Mat residualImage = Mat::zeros(refDepth.rows, refDepth.cols, CV_64FC1);
            MatrixXd residuals = MatrixXd::Zero(refDepth.rows*refDepth.cols, 1);
            MatrixXd jacobians = MatrixXd::Zero(refDepth.rows*refDepth.cols, 6);

            for (int i = 0; i < 10; ++i) {
                odometry.computeResidualsAndJacobians(refGray, refDepth, actGray, residuals, jacobians);
                odometry.optimize(residuals, jacobians);

                for (int i = 0; i < residuals.rows(); ++i) {
                    residualImage.at<double>(i) = residuals(i);
                }

                imshow("Residual", residualImage);
            }            
            cerr << odometry.getMatrixRtFromPose6D() << "\n\n";            
        }
        key = waitKey(1);
        if(key == ' '){
            paused = !paused;
        }
        else if(key == '1'){
            paused = false;
        }
        else if(key == 'q'){
            break;
        }
        viewer.spinOnce(100);
    }

    return 0;
}
