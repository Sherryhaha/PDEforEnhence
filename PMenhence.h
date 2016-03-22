//
// Created by sunguoyan on 16/3/22.
//

#ifndef PMENHENCE_PMENHENCE_H
#define PMENHENCE_PMENHENCE_H

#include <cv.h>
#include <opencv2/highgui.hpp>
using namespace cv;
class PMenhence {

//public:
//    PMenhence();


public:

    unsigned int X_image, Y_image;
//    BYTE *our_image_buffer;
//    BYTE *temp_imageBuffer;
//
//    CTreeCtrl m_tree;
//    HTREEITEM m_hRoot;
//    CEdit m_wndEdit;
//    CCoolBar m_wndMyBar1;
//    CCoolBar m_wndMyBar2;

//    virtual ~CMainFrame();

    int m_nDWTCurDepth;
    int m_nSupp;


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
