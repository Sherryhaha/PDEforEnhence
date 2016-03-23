#include <iostream>
#include "PMenhence.h"
#include<math.h>

using namespace std;
using namespace cv;

typedef double ty;

void PMenhence::gradn(double*A,double*B)    //求N方向梯度
{
    int h, w;
    for (h = 1; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
//            B.at<ty>(h, w) = A.at<ty>(h - 1, w) - A.at<ty>(h, w);
            B[h*X_image+w] = A[(h-1)*X_image+w]-A[h*X_image+w];
        }
}

void PMenhence::grads(double*A, double*B)    //求S方向梯度
{
    int h, w;
    for (h = 0; h < Y_image - 1; h++)
        for (w = 0; w < X_image; w++) {
//            B.at<ty>(h, w) = A.at<ty>(h + 1, w) - A.at<ty>(h, w);
            B[h*X_image+w] = A[(h+1)*X_image+w]-A[h*X_image+w];
        }
}

void PMenhence::grade(double*A, double*B)    //求E方向梯度
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image - 1; w++) {
//            B.at<ty>(h, w) = A.at<ty>(h, w + 1) - A.at<ty>(h, w);
            B[h*X_image+w] = A[h*X_image+w+1]-A[h*X_image+w];
        }
}

void PMenhence::gradw(double*A, double*B)    //求W方向梯度
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 1; w < X_image; w++) {
//            B.at<ty>(h, w) = A.at<ty>(h, w - 1) - A.at<ty>(h, w);
            B[h*X_image+w] = A[h*X_image+w-1]-A[h*X_image+w];
        }
}

void PMenhence::pm1_diffusion(double*A, double*B,
                              double k)    //按照第一个公式求扩散系数'pm1': perona-malik, c=exp{-(|grad(J)|/K)^2} [PM90]
{
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
//            B.at<ty>(h, w) = exp(-(abs(A.at<ty>(h, w)) / k) * (abs(A.at<ty>(h, w)) / k));
            B[h*X_image+w]=exp(-(abs(A[h*X_image+w])/k)*(abs(A[h*X_image+w])/k));
        }
}

void PMenhence::pm2_diffusion(double*A,double*B,
                              double k)    //按照第二个公式求扩散系数'pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2} [PM90]
{
    int h, w;
    int a = X_image;
    int b = Y_image;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
//            B.at<ty>(h, w) = 1 / ((abs(A.at<ty>(h, w)) / k) * (abs(A.at<ty>(h, w)) / k) + 1);
            B[h*X_image+w]=1/((abs(A[h*X_image+w])/k)*(abs(A[h*X_image+w])/k)+1);
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
    double *img;
    img = new double[X_image*Y_image*4];
    for(h = 0;h<Y_image;h++){
        for(w = 0;w<X_image;w++){
            img[h*X_image+w]=src.at<ty>(h,w);
        }
    }


    double * IMG;

    double * pIN;
    double * IS;
    double * IW;
    double * IE;
    double * CN;
    double * CS;
    double * CW;
    double * CE;

    IMG = new double[X_image* Y_image];
    pIN = new double[X_image* Y_image];
    IS = new double[X_image* Y_image];
    IW = new double[X_image* Y_image];
    IE = new double[X_image* Y_image];
    CN = new double[X_image* Y_image];
    CS = new double[X_image* Y_image];
    CW = new double[X_image* Y_image];
    CE = new double[X_image* Y_image];
//    Mat IMG, pIN, IS, IW, IE, CN, CS, CW, CE;
//    IMG.create(src.size(), src.type());
//    pIN.create(src.size(), src.type());
//    IS.create(src.size(), src.type());
//    IW.create(src.size(), src.type());
//    IE.create(src.size(), src.type());
//    CN.create(src.size(), src.type());
//    CS.create(src.size(), src.type());
//    CW.create(src.size(), src.type());
//    CE.create(src.size(), src.type());


    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++)
//            IMG.at<ty>(h, w) = src.at<ty>(h , w );
//            IMG.at<ty>(h, w) = src.at<ty>(h * 4, w * 4);
//            IMG[h*X_image+w]= img[h*X_image*4+w*4];
            IMG[h*X_image+w]= img[h*X_image+w];

//    cout<< src.at<ty>(1000,1000)<<"  ";



    gradn(IMG, pIN);       //求N方向的梯度
    grads(IMG, IS);        //求S方向的梯度
    grade(IMG, IE);        //求E方向的梯度
    gradw(IMG, IW);        //求W方向的梯度

//    namedWindow("testforimg");
//    imshow("testforimg",src);


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
//                in = IMG.at<ty>(h - 1, w) - IMG.at<ty>(h, w);
//                is = IMG.at<ty>(h + 1, w) - IMG.at<ty>(h, w);
//                iw = IMG.at<ty>(h, w - 1) - IMG.at<ty>(h, w);
//                ie = IMG.at<ty>(h, w + 1) - IMG.at<ty>(h, w);
                in = IMG[(h-1)*X_image+w]-IMG[h*X_image+w];
                is = IMG[(h+1)*X_image+w]-IMG[h*X_image+w];
                iw = IMG[h*X_image+w-1]-IMG[h*X_image+w];
                ie = IMG[h*X_image+w+1]-IMG[h*X_image+w];

                switch (feed)        //选择扩散系数是否接受反馈的数据更新
                {
                    case 0:        //不接受反馈数据更新
//                        sum = in * CN.at<ty>(h, w) + is * CS.at<ty>(h, w) + iw * CW.at<ty>(h, w) +
//                              ie * CE.at<ty>(h, w);
                        sum = in*CN[h*X_image+w] + is*CS[h*X_image+w] + iw*CW[h*X_image+w] + ie*CE[h*X_image+w];
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
//                IMG.at<ty>(h, w) = IMG.at<ty>(h, w) + sum;//更新图像数据
                IMG[h*X_image+w]=IMG[h*X_image+w]+sum;
            }
    }

//    cout<<X_image<<" "<<Y_image<<endl;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
//            src.at<ty>(h , w ) = IMG.at<ty>(h, w);
//            src.at<ty>(h * 4, w * 4) = IMG.at<ty>(h, w);
//            src.at<ty>(h * 4, w * 4 + 1) = IMG.at<ty>(h, w);
//            src.at<ty>(h * 4, w * 4 + 2) = IMG.at<ty>(h, w);
//            img[h*X_image*4+w*4]=IMG[h*X_image+w];
//            img[h*X_image*4+w*4+1]=IMG[h*X_image+w];
//            img[h*X_image*4+w*4+2]=IMG[h*X_image+w];
            img[h*X_image+w]=IMG[h*X_image+w];

        }

//    cout<<"come here"<<endl;



    for(h = 0;h < Y_image;h++){
        for(w = 0;w < X_image;w++){
            src.at<ty>(h,w) = img[h*X_image+w];
        }
    }

    delete [] IMG;
    delete [] pIN;
    delete [] IS;
    delete [] IW;
    delete [] IE;
    delete [] CN;
    delete [] CS;
    delete [] CW;
    delete [] CE;
    delete [] img;
    return;
}

void printMat(Mat &src) {
    ty tmp;
    int Height = src.cols;
    int Width = src.rows;
    int i, j;
    for (j = 0; j < Height; j++) {
        for (i = 0; i < Width; i++) {
            tmp = src.at<float>(j, i);
            cout << tmp << " ";
        }
        cout << endl;
    }
}

int main() {

    string filename = "/Users/sunguoyan/Downloads/picture/lenazao.bmp";
    Mat src, src1;
    src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

    //float->CV_32F,double->CV_64F
    src.convertTo(src1, CV_64F, 1.0 / 255.0,0);
//    printMat(src);
    PMenhence p;
    p.X_image = src.cols;
    p.Y_image = src.rows;

// 			method:使用的扩散系数算法	0='pm1': perona-malik, c=exp{-(|grad(J)|/K)^2}
//										1='pm2': perona-malik, c=1/{1+(|grad(J)|/K)^2}
    int method = 0;
    //			feed:是否使用计算出的数据反馈更新扩散系数	0=不更新
    // 														1=更新

    int feed = 1;
//    loop:叠代循环的次数
    int loop = 20;
//    K:扩散系数的压迫因子
    double k = 15;
//    dt:时间增量参数
    double dt = 0.15;
//
//    int method = 1;
//    int feed = 0;
//    int loop = 250;
//    double K = 1;
//    double dt = 0.25;
    p.DiffusionPic(src1, method, feed, loop, k, dt);


    namedWindow("test");
    namedWindow("yuantu");

    imshow("test", src1);
    imshow("yuantu",src);

    waitKey(0);
    return 0;



}