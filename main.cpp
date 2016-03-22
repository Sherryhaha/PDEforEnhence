#include <iostream>
#include "PMenhence.h"
#include<math.h>
#include <cv.h>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void PMenhence::gradn(Mat &A, Mat &B)    //求N方向梯度
{
    int h, w;
    for (h = 1; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
            B.at<uchar>(h, w) = A.at<uchar>(h - 1, w) - A.at<uchar>(h, w);
//            B[h*X_image+w] = A[(h-1)*X_image+w]-A[h*X_image+w];
        }
}

void PMenhence::grads(Mat &A, Mat &B)    //求S方向梯度
{
    int h, w;
    for (h = 0; h < Y_image - 1; h++)
        for (w = 0; w < X_image; w++) {
            B.at<uchar>(h, w) = A.at<uchar>(h + 1, w) - A.at<uchar>(h, w);
//            B[h*X_image+w] = A[(h+1)*X_image+w]-A[h*X_image+w];
        }
}

void PMenhence::grade(Mat &A, Mat &B)    //求E方向梯度
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image - 1; w++) {
            B.at<uchar>(h, w) = A.at<uchar>(h, w + 1) - A.at<uchar>(h, w);
//            B[h*X_image+w] = A[h*X_image+w+1]-A[h*X_image+w];
        }
}

void PMenhence::gradw(Mat &A, Mat &B)    //求W方向梯度
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 1; w < X_image; w++) {
            B.at<uchar>(h, w) = A.at<uchar>(h, w - 1) - A.at<uchar>(h, w);
//            B[h*X_image+w] = A[h*X_image+w-1]-A[h*X_image+w];
        }
}

void PMenhence::pm1_diffusion(Mat &A, Mat &B,
                              double k)    //按照第一个公式求扩散系数'pm1': perona-malik, c=exp{-(|grad(J)|/K)^2} [PM90]
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
            B.at<uchar>(h, w) = exp(-(abs(A.at<uchar>(h, w)) / k) * (abs(A.at<uchar>(h, w)) / k));
//            B[h*X_image+w]=exp(-(abs(A[h*X_image+w])/k)*(abs(A[h*X_image+w])/k));
        }
}

void PMenhence::pm2_diffusion(Mat &A, Mat &B,
                              double k)    //按照第二个公式求扩散系数'pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2} [PM90]
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
            B.at<uchar>(h, w) = 1 / ((abs(A.at<uchar>(h, w)) / k) * (abs(A.at<uchar>(h, w)) / k) + 1);
//            B[h*X_image+w]=1/((abs(A[h*X_image+w])/k)*(abs(A[h*X_image+w])/k)+1);
        }
}




//***********************************************************************************//
// 函数名称: DiffusionPic(img, method, feed, loop, K, dt, Bmp_head)
// 函数功能: 对图像进行Perona-Malik 扩散
// 参数说明:
//			img:输入图像（以矩阵形式）
//			method:使用的扩散系数算法	0='pm1': perona-malik, c=exp{-(|grad(J)|/K)^2}
//										1='pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2}
//			feed:是否使用计算出的数据反馈更新扩散系数	0=不更新
//														1=更新
//			loop:叠代循环的次数
//			K:扩散系数的压迫因子
//			dt:时间增量参数
//			Bmp_head:24位BMP文件的文件头，用于图像文件输出
//
//***********************************************************************************//

void PMenhence::DiffusionPic(Mat &src, int method, int feed, int loop, double K, double dt) {
    int h, w;
    int n = 0;
    double is = 0, in = 0, iw = 0, ie = 0, cn = 0, cs = 0, cw = 0, ce = 0, sum = 0;

//    double * IMG;
//
//    double * pIN;
//    double * IS;
//    double * IW;
//    double * IE;
//    double * CN;
//    double * CS;
//    double * CW;
//    double * CE;
//
//    IMG = new double[X_image* Y_image];
//    pIN = new double[X_image* Y_image];
//    IS = new double[X_image* Y_image];
//    IW = new double[X_image* Y_image];
//    IE = new double[X_image* Y_image];
//    CN = new double[X_image* Y_image];
//    CS = new double[X_image* Y_image];
//    CW = new double[X_image* Y_image];
//    CE = new double[X_image* Y_image];
    Mat IMG, pIN, IS, IW, IE, CN, CS, CW, CE;
    IMG.create(src.size(), src.type());
    pIN.create(src.size(), src.type());
    IS.create(src.size(), src.type());
    IW.create(src.size(), src.type());
    IE.create(src.size(), src.type());
    CN.create(src.size(), src.type());
    CS.create(src.size(), src.type());
    CW.create(src.size(), src.type());
    CE.create(src.size(), src.type());

    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++)
            IMG.at<uchar>(h, w) = src.at<uchar>(h * 4, w * 4);
//            IMG[h*X_image+w]= img[h*X_image*4+w*4];


    gradn(IMG, pIN);    //求N方向的梯度
    grads(IMG, IS);        //求S方向的梯度
    grade(IMG, IE);        //求E方向的梯度
    gradw(IMG, IW);        //求W方向的梯度

    switch (method) {
        case 0:        //按照第一个公式求各个方向的扩散系数'pm1': perona-malik, c=exp{-(|grad(J)|/K)^2} [PM90]
            pm1_diffusion(pIN, CN, K);
            pm1_diffusion(IS, CS, K);
            pm1_diffusion(IW, CW, K);
            pm1_diffusion(IE, CE, K);
            break;
        case 1:        //按照第二个公式求各个方向的扩散系数'pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2} [PM90]
            pm2_diffusion(pIN, CN, K);
            pm2_diffusion(IS, CS, K);
            pm2_diffusion(IW, CW, K);
            pm2_diffusion(IE, CE, K);
            break;
        default:
            return;
    }

    for (n = 0; n < (loop); n++)        //叠代计算
    {
        for (h = 1; h < Y_image - 1; h++)        //计算各个方向梯度
            for (w = 1; w < X_image - 1; w++) {
                in = IMG.at<uchar>(h - 1, w) - IMG.at<uchar>(h, w);
                is = IMG.at<uchar>(h + 1, w) - IMG.at<uchar>(h, w);
                iw = IMG.at<uchar>(h, w - 1) - IMG.at<uchar>(h, w);
                ie = IMG.at<uchar>(h, w + 1) - IMG.at<uchar>(h, w);
//                in = IMG[(h-1)*X_image+w]-IMG[h*X_image+w];
//                is = IMG[(h+1)*X_image+w]-IMG[h*X_image+w];
//                iw = IMG[h*X_image+w-1]-IMG[h*X_image+w];
//                ie = IMG[h*X_image+w+1]-IMG[h*X_image+w];

                switch (feed)        //选择扩散系数是否接受反馈的数据更新
                {
                    case 0:        //不接受反馈数据更新
                        sum = in * CN.at<uchar>(h, w) + is * CS.at<uchar>(h, w) + iw * CW.at<uchar>(h, w) +
                              ie * CE.at<uchar>(h, w);
//                        sum = in*CN[h*X_image+w] + is*CS[h*X_image+w] + iw*CW[h*X_image+w] + ie*CE[h*X_image+w];
                        break;
                    case 1:        //接受反馈数据更新
                        cn = 1 / (1 + (abs(in) / K) * (abs(in) / K));
                        cs = 1 / (1 + (abs(is) / K) * (abs(is) / K));
                        cw = 1 / (1 + (abs(iw) / K) * (abs(iw) / K));
                        ce = 1 / (1 + (abs(ie) / K) * (abs(ie) / K));

                        sum = in * cn + is * cs + iw * cw + ie * ce;
                        break;
                    default:
                        return;
                }

                sum = sum * dt;        //乘时间增量因子
                IMG.at<uchar>(h, w) = IMG.at<uchar>(h, w) + sum;//更新图像数据
//                IMG[h*X_image+w]=IMG[h*X_image+w]+sum;
            }
    }

    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
            src.at<uchar>(h * 4, w * 4) = IMG.at<uchar>(h, w);
            src.at<uchar>(h * 4, w * 4 + 1) = IMG.at<uchar>(h, w);
            src.at<uchar>(h * 4, w * 4 + 2) = IMG.at<uchar>(h, w);
//            img[h*X_image*4+w*4]=IMG[h*X_image+w];
//            img[h*X_image*4+w*4+1]=IMG[h*X_image+w];
//            img[h*X_image*4+w*4+2]=IMG[h*X_image+w];
        }
//
//    delete [] IMG;
//    delete [] pIN;
//    delete [] IS;
//    delete [] IW;
//    delete [] IE;
//    delete [] CN;
//    delete [] CS;
//    delete [] CW;
//    delete [] CE;
}

void printMat(Mat &src) {
    unsigned char tmp;
    int Height = src.cols;
    int Width = src.rows;
    int i, j;
    for (j = 0; j < Height; j++) {
        for (i = 0; i < Width; i++) {
            tmp = src.at<uchar>(j, i);
            cout << tmp << " ";
        }
        cout << endl;
    }
}

int main() {

    string filename = "/Users/sunguoyan/Downloads/picture/wu.jpg";
    Mat src, src1;
    src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

//    src.convertTo(src1, CV_32F, 1 / 255.0);
//    printMat(src);
    PMenhence p;
    p.X_image = src.rows;
    p.Y_image = src.cols;

// 			method:使用的扩散系数算法	0='pm1': perona-malik, c=exp{-(|grad(J)|/K)^2}
//										1='pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2}
    int method = 0;
    //			feed:是否使用计算出的数据反馈更新扩散系数	0=不更新
    // 														1=更新

    int feed = 0;
//    loop:叠代循环的次数
    int loop = 3;
//    K:扩散系数的压迫因子
    double k = 1;
//    dt:时间增量参数
    double dt = 1;

    p.DiffusionPic(src, method, feed, loop, k, dt);


    namedWindow("test");
    imshow("test", src);
    waitKey(0);



}