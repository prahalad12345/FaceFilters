#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include<vector>

/*
    removing blemishes from the face using seamless cloning
*/
using namespace std;
using namespace cv;

Point prevPt(-1,-1);
Mat bgr;
Mat skin(Size(15,15),CV_8UC3);
Mat mask;

void onMouse(int event,int x,int y,int flag,void*){
    //using mouse click detect the region of skin on which the mouse is being clicked 
    if(event == EVENT_LBUTTONDOWN){
        prevPt=Point(x,y);
        if(x-20>=1 && y-20>=1 && x<bgr.size().width && y<bgr.size().height)
        	seamlessClone(skin,bgr,mask,prevPt,bgr,NORMAL_CLONE);
        //seamless cloning the white pixel on the skin
    }
        imshow("I Remove Blemish",bgr);
}

int main(){
    bgr=imread("./images/1.png");//input file containing the blemish

    namedWindow("I Remove Blemish");

    skin.setTo(Scalar(255,255,255));//the background which has to be replaced on the skin 
    mask=skin.clone();
    imshow("I Remove Blemish",bgr);
    setMouseCallback("I Remove Blemish",onMouse,NULL);//calling the mouse click callback function
    while(1){
        char c=(char)waitKey();
        if(c=='q')
            break;
    }
    
    waitKey(0);
    destroyAllWindows();
    return 0;
}
