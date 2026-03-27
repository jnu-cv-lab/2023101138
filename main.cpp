#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
    // ====================== mission1——读取图片 ======================
    string img_path = "/mnt/e/amnesiac/Pictures/lkl/IMG_20241128_154344.jpg";
    Mat img = imread(img_path);

    if (img.empty()) {
        cerr << "无法读取图片" << endl;
        return -1;
    } else {
        cout << "图片读取成功" << endl;
    }

    // ====================== mission2——输出图片基本信息 ======================
    int H = img.rows;
    int W = img.cols;
    int Channels = img.channels();
    string photo_type = img.type() == CV_8UC3 ? "CV_8UC3" : "其他类型";

    cout << "图像宽度 = " << W << "，图像高度 = " << H << endl;
    cout << "图像通道数 = " << Channels << endl;
    cout << "图像类型 = " << photo_type << endl;

    // ====================== mission3——保存原图（BGR转RGB） ======================
    Mat img_rgb;
    cvtColor(img, img_rgb, COLOR_BGR2RGB);
    imwrite("original_photo.png", img);  // 直接保存原图

    // ====================== mission4——转为灰度图 ======================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    // ====================== mission5——保存灰度图 ======================
    imwrite("gray_result.png", gray_img);
    cout << "灰图已保存" << endl;

    // ====================== mission6——输出左上角像素值 ======================
    Vec3b photo_val = img_rgb.at<Vec3b>(0, 0);
    cout << "左上角像素(0,0)的RGB值：" 
         << (int)photo_val[0] << " " 
         << (int)photo_val[1] << " " 
         << (int)photo_val[2] << endl;

    return 0;
    
}