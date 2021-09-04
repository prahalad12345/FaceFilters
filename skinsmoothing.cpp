#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

#define CV_HAAR_SCALE_IMAGE 2
/*
    detecting the skin tone using a pixel from the forehead and using it as color code to detect the skin pixel and removing the unnecessary background
    
*/
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat img=imread("../images/2.jpg");

    //reading the haar files
    string faceCascadeName = "face.xml";

    CascadeClassifier faceCascade;
    faceCascade.load(faceCascadeName) ;
    
    
    Mat frameGray;

    vector<Rect> faces;//vertices of the face region
    vector<Point> centers;
    faceCascade.detectMultiScale(img, faces, 1.1, 2, 0|2, Size(30, 30) );
    Mat hsvimage,faceROI;
    //comparing with hsv image is easier hence converting it and finding the range of color satisfying it
    //faceROI variable stores the face region
    Vec3b huecolor;
    //hue color of the skin on the forehead
    for(int i=0;i<faces.size();i++){
        cout<<faces[i]<<endl;
        faceROI = img(faces[i]);
        cvtColor(faceROI,hsvimage,COLOR_BGR2HSV);
        Point skin=Point((faceROI.size().width)/2,faceROI.size().height/8);
        huecolor=hsvimage.at<Vec3b>(skin);
    }
    
    Vec3b lower={11,70,60};
    Vec3b upper={11,70,60};
    lower=huecolor-lower;
    
    Mat resultskin,skin;
    //skin stores the region where the skin is present acts like a mask for the skin
    Mat notskin,resultnotskin;
    //notskin stores the part of the image apart from the skin on the face
    inRange(hsvimage,lower,upper,skin);//inRange function extract the skin region based on their skin pixels

    Mat faceroiclone=faceROI.clone();
    Mat blurface;
    bilateralFilter(faceroiclone,blurface,15,80,90);//to remove all the noises or rough patch from the skin - bilateral blur the image
    
    bitwise_and(blurface,blurface,resultskin,skin);//resultskin variable is obtained by using a mask on the blurface variable
    bitwise_not(skin,notskin);
    bitwise_and(faceROI,faceROI,resultnotskin,notskin);//obtain the background

    Mat finalimage;
    add(resultskin,resultnotskin,finalimage);
    //finalimage stores the filtered face

    Mat newframe=img.clone();
    finalimage.copyTo(newframe(faces[0]));

    waitKey(0);
    destroyAllWindows();
    return 0;
}
