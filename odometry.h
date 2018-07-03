#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <eigen3/Eigen/Core>
#include <pcl/common/common.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "projector.h"

using namespace std;
using namespace cv;
using namespace Eigen;

class Odometry{

public:

    VectorXd poseVector6D;
    Projector projector;

    Mat sourceIntensityImage;
    Mat sourceDepthImage;
    Mat targetIntensityImage;
    Mat targetDepthImage;

    Odometry(Projector projector){
        poseVector6D.setZero(6);
        this->projector = projector;
    }

    VectorXd getPose6D(){
        return poseVector6D;
    }

    void setsourceRGBImages(Mat sourceIntensityImage, Mat sourceDepthImage){
        this->sourceIntensityImage = sourceIntensityImage;
        this->sourceDepthImage = sourceDepthImage;
    }

    void setsourceDepthImages(Mat targetIntensityImage, Mat targetDepthImage){
        this->targetIntensityImage = targetIntensityImage;
        this->targetDepthImage = targetDepthImage;
    }

    void setInitialPoseVector(VectorXd poseVector6D){
        this->poseVector6D = poseVector6D;
    }

    void computeResidualsAndJacobians(Mat &sourceIntensityImage, Mat &sourceDepthImage,
                                      Mat &targetIntensityImage,
                                      MatrixXd &residuals, MatrixXd &jacobians){

        Mat targetDerivativeX, targetDerivativeY;

        Scharr(targetIntensityImage, targetDerivativeX, CV_64F, 1, 0, 0.00005, 0.0, cv::BORDER_DEFAULT);
        Scharr(targetIntensityImage, targetDerivativeY, CV_64F, 0, 1, 0.00005, 0.0, cv::BORDER_DEFAULT);

        double x = poseVector6D(0);
        double y = poseVector6D(1);
        double z = poseVector6D(2);
        double yaw = poseVector6D(3);
        double pitch = poseVector6D(4);
        double roll = poseVector6D(5);


        //*********** Compute the rigid transformation matrix from the poseVector6D ************//
        Matrix4d Rt = Matrix4d::Identity();
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double sin_pitch = sin(pitch);
        double cos_pitch = cos(pitch);
        double sin_roll = sin(roll);
        double cos_roll = cos(roll);
        Rt(0,0) = cos_yaw * cos_pitch;
        Rt(0,1) = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
        Rt(0,2) = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
        Rt(0,3) = x;
        Rt(1,0) = sin_yaw * cos_pitch;
        Rt(1,1) = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
        Rt(1,2) = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
        Rt(1,3) = y;
        Rt(2,0) = -sin_pitch;
        Rt(2,1) = cos_pitch * sin_roll;
        Rt(2,2) = cos_pitch * cos_roll;
        Rt(2,3) = z;
        Rt(3,0) = 0.0;
        Rt(3,1) = 0.0;
        Rt(3,2) = 0.0;
        Rt(3,3) = 1.0;

        //*********** Precomputation to estimate jacobians ************//

        double temp1 = cos(pitch)*sin(roll);
        double temp2 = cos(pitch)*cos(roll);
        double temp3 = sin(pitch);
        double temp4 = (sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw));
        double temp5 = (sin(pitch)*sin(roll)*cos(yaw)-cos(roll)*sin(yaw));
        double temp6 = (sin(pitch)*sin(roll)*sin(yaw)+cos(roll)*cos(yaw));
        double temp7 = (-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw));
        double temp8 = (sin(roll)*cos(yaw)-sin(pitch)*cos(roll)*sin(yaw));
        double temp9 = (sin(pitch)*cos(roll)*sin(yaw)-sin(roll)*cos(yaw));
        double temp10 = cos(pitch)*sin(roll)*cos(yaw);
        double temp11 = cos(pitch)*cos(yaw)+x;
        double temp12 = cos(pitch)*cos(roll)*cos(yaw);
        double temp13 = sin(pitch)*cos(yaw);
        double temp14 = cos(pitch)*sin(yaw);
        double temp15 = cos(pitch)*cos(yaw);
        double temp16 = sin(pitch)*sin(roll);
        double temp17 = sin(pitch)*cos(roll);
        double temp18 = cos(pitch)*sin(roll)*sin(yaw);
        double temp19 = cos(pitch)*cos(roll)*sin(yaw);
        double temp20 = sin(pitch)*sin(yaw);
        double temp21 = (cos(roll)*sin(yaw)-sin(pitch)*sin(roll)*cos(yaw));
        double temp22 = cos(pitch)*cos(roll);
        double temp23 = cos(pitch)*sin(roll);
        double temp24 = cos(pitch);

        //*********** Unprojection, projection ************//
        double fx = projector.fx;
        double fy = projector.fy;
        double centerX = projector.centerX;
        double centerY = projector.centerY;

        int nRows = sourceDepthImage.rows;
        int nCols = sourceDepthImage.cols;

//        #pragma omp parallel for
        for (int r = 0; r < nRows; ++r) {
            for (int c = 0; c < nCols; ++c) {

                //******* BEGIN Unprojection of DepthMap ********//
                Vector4d point3D;
                point3D(2) = *sourceDepthImage.ptr<ushort>(r, c)/5000.f;
                if(point3D(2) == 0)
                    continue;
                point3D(0) = (c - centerX) * point3D(2) * 1/fx;
                point3D(1) = (r - centerY) * point3D(2) * 1/fy;
                point3D(3) = 1.0;

                double px = point3D(0);
                double py = point3D(1);
                double pz = point3D(2);
                //******* END Unprojection of DepthMap ********//

                //******* BEGIN Transformation of PointCloud ********//
                Vector4d transfPoint3D = Rt * point3D;
                //******* END Transformation of PointCloud ********//

                //******* BEGIN Projection of PointCloud on the image plane ********//
                double invTransfZ = 1.0/transfPoint3D(2);
                double transfC = (transfPoint3D(0) * fx) * invTransfZ + centerX;
                double transfR = (transfPoint3D(1) * fy) * invTransfZ + centerY;
                int transfR_int = static_cast<int>(round( transfR ));
                int transfC_int = static_cast<int>(round( transfC ));
                //******* END Projection of PointCloud on the image plane ********//

                //******* BEGIN Residual and Jacobians computation ********//
                //Checks if this pixel projects inside of the source image
                if((transfR_int >= 0 && transfR_int < nRows) &
                   (transfC_int >= 0 && transfC_int < nCols)) {

                    double pixel1 = *sourceIntensityImage.ptr<uchar>(r, c)/255.f;
                    double pixel2 = *targetIntensityImage.ptr<uchar>(transfR_int, transfC_int)/255.f;

                    //Compute the per pixel jacobian
                    MatrixXd jacobianPrRt(2,6);
                    double temp25 = 1.0/(z + py * temp1 + pz * temp2 - px * temp3);
                    double temp26 = temp25*temp25;

                    //Derivative with respect to x
                    jacobianPrRt(0,0) = fx*temp25;
                    jacobianPrRt(1,0) = 0.0;

                    //Derivative with respect to y
                    jacobianPrRt(0,1) = 0.0;
                    jacobianPrRt(1,1) = fy*temp25;

                    //Derivative with respect to z
                    jacobianPrRt(0,2) = -fx*(pz*temp4+py*temp5+px*temp11)*temp26;
                    jacobianPrRt(1,2) = -fy*(py*temp6+pz*temp9+px*temp14+y)*temp26;

                    //Derivative with respect to yaw
                    jacobianPrRt(0,3) = fx*(py*temp7+pz*temp8-px*temp14)*temp25;
                    jacobianPrRt(1,3) = fy*(pz*temp4+py*temp5+px*temp15)*temp25;

                    //Derivative with respect to pitch
                    jacobianPrRt(0,4) = fx*(py*temp10+pz*temp12-px*temp13)*temp25
                                        -fx*(-py*temp16-pz*temp17-px*temp24)*(pz*temp4+py*temp5+px*temp11)*temp26;
                    jacobianPrRt(1,4) = fy*(py*temp18+pz*temp19-px*temp20)*temp25
                                        -fy*(-py*temp16-pz*temp17-px*temp24)*(py*temp6+pz*temp9+px*temp14+y)*temp26;

                    //Derivative with respect to roll
                    jacobianPrRt(0,5) = fx*(py*temp4+pz*temp21)*temp25
                                        -fx*(py*temp22-pz*temp23)*(pz*temp4+py*temp5+px*temp11)*temp26;
                    jacobianPrRt(1,5) = fy*(pz*temp7+py*temp9)*temp25
                                        -fy*(py*temp22-pz*temp23)*(py*temp6+pz*temp9+px*temp14+y)*temp26;

                    //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
                    MatrixXd targetGradient(1,2);
                    uint i = nCols * r + c;
                    targetGradient(0) = *targetDerivativeX.ptr<double>(r,c);
                    targetGradient(1) = *targetDerivativeY.ptr<double>(r,c);
                    MatrixXd jacobian = targetGradient * jacobianPrRt;

                    //Assign the pixel residual and jacobian to its corresponding row
                    jacobians(i,0) = jacobian(0,0);
                    jacobians(i,1) = jacobian(0,1);
                    jacobians(i,2) = jacobian(0,2);
                    jacobians(i,3) = jacobian(0,3);
                    jacobians(i,4) = jacobian(0,4);
                    jacobians(i,5) = jacobian(0,5);
                    //Residual of the pixel
                    residuals(nCols * transfR_int + transfC_int, 0) = pixel2 - pixel1;
                }
                //******* END Residual and Jacobians computation ********//
            }
        }
    }

    Matrix4f getMatrixRtFromPose6D(){
        //*********** Compute the rigid transformation matrix from the poseVector6D ************//
        Matrix4f Rt = Matrix4f::Zero();
        double x = poseVector6D(0);
        double y = poseVector6D(1);
        double z = poseVector6D(2);
        double yaw = poseVector6D(3);
        double pitch = poseVector6D(4);
        double roll = poseVector6D(5);

        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double sin_pitch = sin(pitch);
        double cos_pitch = cos(pitch);
        double sin_roll = sin(roll);
        double cos_roll = cos(roll);
        Rt(0,0) = cos_yaw * cos_pitch;
        Rt(0,1) = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
        Rt(0,2) = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
        Rt(0,3) = x;
        Rt(1,0) = sin_yaw * cos_pitch;
        Rt(1,1) = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
        Rt(1,2) = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
        Rt(1,3) = y;
        Rt(2,0) = -sin_pitch;
        Rt(2,1) = cos_pitch * sin_roll;
        Rt(2,2) = cos_pitch * cos_roll;
        Rt(2,3) = z;
        Rt(3,0) = 0.0;
        Rt(3,1) = 0.0;
        Rt(3,2) = 0.0;
        Rt(3,3) = 1.0;

        return Rt;
    }

    void optimize(MatrixXd &residuals, MatrixXd &jacobians){

        MatrixXd gradients = jacobians.transpose() * residuals;
        poseVector6D = poseVector6D - ((jacobians.transpose()*jacobians).inverse() * gradients);
        cerr << gradients.norm() << endl;
    }
};
#endif
