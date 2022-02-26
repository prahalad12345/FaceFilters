#include "faceBlendCommon.hpp"
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace dlib;
using namespace std;
using namespace cv;



int main(){
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkdetector;
    deserialize("shape_predictor_68_face_landmarks.dat") >> landmarkdetector;
    Mat img1=imread("girl-no-makeup.jpg");
    imshow("initial",img1);
    cv_image<bgr_pixel> dlibIm(img1);
    std::vector<dlib::rectangle> facerect=faceDetector(dlibIm);
    cout<<facerect.size()<<endl;
    full_object_detection landmarks=landmarkdetector(dlibIm,facerect[0]);
    std::vector<Point2f> point1,point2;
    for(int i=0;i<68;i++){
        point1.push_back(Point(landmarks.part(i).x(),landmarks.part(i).y()));
    }
    //the above lines involve extraction of 68 points using dlib

    //initializing cheek color to be red
    Mat cheekcolor=Mat::zeros(img1.size().height,img1.size().width,CV_8UC3);
    cheekcolor.setTo(Scalar(0,0,255));
    Mat mask=Mat::zeros(img1.size().height,img1.size().width,CV_8UC3);
    //drawing the mask after finding the centroid of each cheek and drawing a circle
    //trying to implement a simple circle blush
    Point center=Point((point1[0].x+point1[30].x+point1[4].x)/3,(point1[4].y+point1[1].y+point1[30].y)/3);
    cv::circle(mask,center,(point1[31].x-point1[2].x)/3,(255,0,255),-1);
    

    Point center1=Point((point1[12].x+point1[35].x+point1[16].x)/3,(point1[12].y+point1[35].y+point1[16].y)/3);
    cv::circle(mask,center1,(point1[14].x-point1[35].x)/3,(255,255,255),-1);
    Mat masks[3];
    
    split(mask,masks);
    Mat cheeks;
    //obtained mask in 1 channel
    bitwise_and(cheekcolor,cheekcolor,cheeks,masks[0]);
    
    //above code extracts the cheek of the specific color
    //here chosen red

    GaussianBlur(cheeks,cheeks,Size(9,9),0,0);
    Mat result;
    //adding a lower weight for the cheeks image on implementing with higher weight it gives an outcasting circle making it look unrealistic

    addWeighted(img1,1,cheeks,0.05,0,result);
    
    imshow("result",result);
    waitKey(0);
}
