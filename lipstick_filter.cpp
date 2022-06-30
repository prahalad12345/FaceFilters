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

Mat img;
Mat lipcolor;
int col1=0;
Mat lip1;
Mat masksoff[3];
Mat lipmask;
Mat eightu;
Mat result;
Scalar colors[5]={Scalar(75,0,130),Scalar(0,0,255),Scalar(255,0,255),Scalar(128,0,128),Scalar(153,50,204)};
/*
    Shades of pink and Red color 
*/

void lipchoice(int,void*){
    lipcolor.setTo(colors[col1]);
    eightu=masksoff[0];
    bitwise_and(lipcolor,lipcolor,lip1,eightu);
    //reading the image every time the slider is called and applying the lip flter
    img=imread("girl-no-makeup.jpg");
    
    GaussianBlur(lip1,lip1,Size(5,5),0,0);
    //blur the image add weight and show result
    addWeighted(img,1,lip1,0.95,0,result);
    imshow("result",result);
}


void removepolygonfrommask(Mat &mask,std::vector<Point2f> points,std::vector<int>Pointindex){
    //extract the points of the convex create a hull and fill it with (0,0,0) indicating empty par of mask
    std::vector<Point> hullpoints;
    for(int i=0;i<Pointindex.size();i++){
        Point pt(points[Pointindex[i]].x,points[Pointindex[i]].y);
        hullpoints.push_back(pt);
    }
    fillConvexPoly(mask,&hullpoints[0],hullpoints.size(),Scalar(0,0,0));
}

Mat getlipmask(Size size,std::vector<Point2f> points){
    static int outerlips[]={48,49,50,51,52,53,54,55,56,57,58,59};
    std::vector<int> outerlipsindex (outerlips,outerlips+sizeof(outerlips)/sizeof(outerlips[0]));
    static int innerlips[]= {60,61,62,63,64,65,66,67};
    std::vector<int> innerlipsindex (innerlips,innerlips+sizeof(innerlips)/sizeof(innerlips[0]));
    std::vector<Point2f> hull;
    
    cout<<size.height<<" "<<size.width<<endl;
    std::vector<Point2f> mouth;
    for(int i=0;i<outerlipsindex.size();i++)
        mouth.push_back(points[outerlipsindex[i]]);
    std::vector<Point> hullint;
    convexHull(mouth,hull,false,true);
    for(int i=0;i<hull.size();i++){
        Point pt(hull[i].x,hull[i].y);
        hullint.push_back(pt);
    }
    Mat mask=Mat::zeros(size.height,size.width,CV_8UC3);
    fillConvexPoly(mask,&hullint[0],hullint.size(),Scalar(255,255,255));

    removepolygonfrommask(mask,points,innerlipsindex);
    return mask;
}


int main(){
    //final window to be shown
    namedWindow("result",WINDOW_AUTOSIZE);
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkdetector;
    deserialize("shape_predictor_68_face_landmarks.dat") >> landmarkdetector;
    Mat img=imread("girl-no-makeup.jpg");
    cv_image<bgr_pixel> dlibIm(img);
    std::vector<dlib::rectangle> facerect=faceDetector(dlibIm);
    
    full_object_detection landmarks=landmarkdetector(dlibIm,facerect[0]);
    std::vector<Point2f> face;
    for(int i=0;i<68;i++){
        face.push_back(Point(landmarks.part(i).x(),landmarks.part(i).y()));
    }
    //extracting the 68 points
    lipcolor=Mat::zeros(img.size().height,img.size().width,CV_8UC3);
    lipcolor.setTo(colors[0]);
    //first showing color[0]on the lips
    lipmask=getlipmask(img.size(),face);
    imshow("lipmask",lipmask);
    split(lipmask,masksoff);
    eightu=masksoff[0];
    bitwise_and(lipcolor,lipcolor,lip1,eightu);
    //obtaining the lip color in the mask
    
    imshow("lipcolor",lipcolor);
    GaussianBlur(lip1,lip1,Size(5,5),0,0);
    
    addWeighted(img,1,lip1,0.95,0,result);
    imshow("result",result);
    createTrackbar("lipstick choice","result",&col1,4,lipchoice);
    
    waitKey(0);
    return 0;
}
