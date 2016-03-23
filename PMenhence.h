//
// Created by sunguoyan on 16/3/22.
//

#ifndef PMENHENCE_PMENHENCE_H
#define PMENHENCE_PMENHENCE_H

#include <cv.h>
#include <opencv2/highgui.hpp>
using namespace cv;
class PMenhence {



public:

     int X_image, Y_image;


    void gradn(Mat & A, Mat&B);

    void grads(Mat&A, Mat&B);

    void grade(Mat&A, Mat&B);

    void gradw(Mat&A, Mat&B);

    void pm1_diffusion(Mat&A, Mat&B, double k);

    void pm2_diffusion(Mat&A, Mat&B, double k);

    void DiffusionPic(Mat&img, int method, int feed, int loop, double K, double dt);

//    void TVDenoising(double *my_imageBuffer, int iter /* = 80 */);
//
//    double **newMatrix(int nx, int ny);
//
//    bool delMatrix(double **matrix, int nx, int ny);

};

#endif //PMENHENCE_PMENHENCE_H
