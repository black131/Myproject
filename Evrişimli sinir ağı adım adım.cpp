#include <cstdlib>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    system("PAUSE");
    return EXIT_SUCCESS;
}
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv; 
using namespace;

Mat zero_pad(Mat X, int pad) {
    Mat X_pad;
    copyMakeBorder(X, X_pad, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
    return X_pad;
}
double conv_single_step(Mat a_slice_prev, Mat W, double b) {
    Mat s;
    multiply(a_slice_prev, W, s);
    double z = sum(s)[0];
    double Z = b + z;
    return Z;
}
vector<Mat> conv_forward(vector<Mat> A_prev, vector<Mat> W, vector<double> b, int stride, int pad) {
    int m = A_prev.size();
    int n_H_prev = A_prev[0].rows;
    int n_W_prev = A_prev[0].cols;
    int n_C_prev = A_prev[0].channels();
    int f = W[0].rows;
    int n_C = W.size();

    int n_H = ((n_H_prev - f + 2 * pad) / stride) + 1;
    int n_W = ((n_W_prev - f + 2 * pad) / stride) + 1;

    vector<Mat> Z(m, Mat::zeros(n_H, n_W, CV_64F));

    for (int i = 0; i < m; ++i) {
        Mat a_prev_pad = zero_pad(A_prev[i], pad);
        for (int h = 0; h < n_H; ++h) {
            for (int w = 0; w < n_W; ++w) {
                for (int c = 0; c < n_C; ++c) {
                    int vert_start = h * stride;
                    int vert_end = vert_start + f;
                    int horiz_start = w * stride;
                    int horiz_end = horiz_start + f;

                    Rect roi(horiz_start, vert_start, f, f);
                    Mat a_slice_prev = a_prev_pad(roi);
                    Z[i].at<double>(h, w) = conv_single_step(a_slice_prev, W[c], b[c]);
                }
            }
        }
    }
    return Z;
}
vector<Mat> pool_forward(vector<Mat> A_prev, int stride, int f, string mode = "max") {
    int m = A_prev.size();
    int n_H_prev = A_prev[0].rows;
    int n_W_prev = A_prev[0].cols;
    int n_C_prev = A_prev[0].channels();

    int n_H = ((n_H_prev - f) / stride) + 1;
    int n_W = ((n_W_prev - f) / stride) + 1;

    vector<Mat> A(m, Mat::zeros(n_H, n_W, CV_64F));

    for (int i = 0; i < m; ++i) {
        for (int h = 0; h < n_H; ++h) {
            for (int w = 0; w < n_W; ++w) {
                for (int c = 0; c < n_C_prev; ++c) {
                    int vert_start = h * stride;
                    int vert_end = vert_start + f;
                    int horiz_start = w * stride;
                    int horiz_end = horiz_start + f;

                    Rect roi(horiz_start, vert_start, f, f);
                    Mat a_prev_slice = A_prev[i](roi);

                    if (mode == "max") {
                        double minVal, maxVal;
                        minMaxLoc(a_prev_slice, &minVal, &maxVal);
                        A[i].at<double>(h, w) = maxVal;
                    } else if (mode == "average") {
                        A[i].at<double>(h, w) = mean(a_prev_slice)[0];
                    }
                }
            }
        }
    }
    return A;
}
