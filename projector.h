#ifndef PROJECTOR_H
#define PROJECTOR_H

#include <pcl/common/common.h>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace pcl;
using namespace cv;

class Projector{

public:
    double fx;
    double fy;
    double centerX;
    double centerY;

    Projector(){

    }

    void setIntrisecs(double fx, double fy, double centerX, double centerY){
        this->fx = fx;
        this->fy = fy;
        this->centerX = centerX;
        this->centerY = centerY;
    }

    void unprojectDepth(Mat &depth, Mat &rgb, PointCloud<PointXYZRGB>::Ptr cloud){

        //        #pragma omp parallel for
        for (int r = 0; r < depth.rows; ++r) {
            ushort* pixelDepth = depth.ptr<ushort>(r);
            Vec3b* pixelRGB = rgb.ptr<Vec3b>(r);
            for (int c = 0; c < depth.cols; ++c) {
                double dep = pixelDepth[c];
                if(dep > 0){
                    double red = pixelRGB[c][2];
                    double gre = pixelRGB[c][1];
                    double blu = pixelRGB[c][0];

                    PointXYZRGB point3D;
                    point3D.z = dep/5000.f;
                    point3D.x = (c - centerX) * point3D.z * 1.f/fx;
                    point3D.y = (r - centerY) * point3D.z * 1.f/fy;
                    point3D.r = red;
                    point3D.g = gre;
                    point3D.b = blu;
                    cloud->points.push_back(point3D);
                }
            }
        }
    }

};
#endif
