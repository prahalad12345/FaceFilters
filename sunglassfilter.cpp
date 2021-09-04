#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

#define CV_HAAR_SCALE_IMAGE 2
/*

using haar cascade to detect the eye region
INSTRUCTIONS:
1.Please set the Reflection of sunglass trackbar To provide the zebra striping on the sunglass
2.Now set the alpha of Bubble trackbar to set the transparency of bubble wrt the glass
3. At last make the changes to the transparency of eyeglass and the region of eye

IMPORTANT NOTE:
1.if transparency of eye and glass is changed before the bubble is changed it will not be detected
2.if the transparency of bubble is change before the transparency of zebra striping image this will not be detected.

In short the order of adding glass to the image:
change "Reflection on the glass"->"alpha of Bubbles"->"Transparency wrt Eye and glass"

Not following the above order leads to repeating the process again

Used frame with an alpha channel for ease

THANK YOU FOR YOUR PATIENCE
*/
using namespace cv;
using namespace std;
Mat frame;//the image of elon musk
Mat blender;//The output from the eyeglass function
Mat maskedEye;//The output after thresholding on the sunglass's alpha mask and sunglass
Mat maskedFrame;//consist of the background of the rectangle frame
Mat blendercv;//stores the result from the eyecv function
Mat opencvimg;//stores the image of the high contrast image
Mat opencvframe;//frame for the reflection image
Mat bubble;//image of the bubble
Mat bubbleframe;//bubble image obtained on the sunglass mask
Mat blendbubble;//final image obtained in the blenderbubble function
Mat eyethreshold;//image of the eye in the sunglass mask
int scalefactor=0,glassscalefactor=0,bubblefactor=0;
int x,w,y,h;

//eyecv is the function dealing with the reflection on the glass

void eyecv(int,void*){
    double blendcv=(double)glassscalefactor/100.0;
    addWeighted(opencvframe,blendcv,maskedEye,1.0-blendcv,0.0,blendercv);
    imshow("blenderercv",blendercv);
    add(blendercv, maskedFrame, frame(Rect(x,y,w,h)));
    imshow("Frame",frame);
}

//blenderbubble is the function which is responsible of dealing with the bubbles on the sunglass frame

void blenderbubble(int ,void *){
    double blendcv=(double)bubblefactor/100.0;
    if(!blendercv.empty())
        addWeighted(bubbleframe,blendcv,blendercv,1.0-blendcv,0.0,blendbubble);
    else 
        addWeighted(bubbleframe,blendcv,maskedEye,1.0-blendcv,0.0,blendbubble);
    imshow("blendererbubble",blendbubble);
    add(blendbubble, maskedFrame, frame(Rect(x,y,w,h)));
    imshow("Frame",frame);
}

//This function deals with the transparency between the glass and the eyes

void eyeglass(int , void *){
    double blend=(double)scalefactor/100.0;
    if(blendbubble.empty() && !blendercv.empty())
        addWeighted(blendercv,blend,eyethreshold,1.0-blend,0.0,blender);   
    else if(blendbubble.empty() && blendercv.empty())
        addWeighted(maskedEye,blend,eyethreshold,1.0-blend,0.0,blender);   
    else
        addWeighted(blendbubble,blend,eyethreshold,1.0-blend,0.0,blender);
    imshow("blenderer",blender);
    add(blender, maskedFrame, frame(Rect(x,y,w,h)));
    imshow("Frame",frame);
}


int main(int argc, char* argv[])
{
    frame=imread("./images/2.jpg");
    namedWindow("Frame",WINDOW_AUTOSIZE);
    imshow("Frame",frame);
    Mat openc=imread("./images/0.jpg");
    bubble=imread("./images/3.jpg");
    Mat eyeMask = imread("./images/sunglass.png");

    cvtColor(openc,openc,COLOR_BGR2GRAY);
    Mat glassRGBchannel[3];
    for(int i=0;i<3;i++)  
        glassRGBchannel[i]=openc;
    merge(glassRGBchannel,3,opencvimg);

    //reading the haar files
    string faceCascadeName = "face.xml";
    string eyeCascadeName = "eye.xml";

    CascadeClassifier faceCascade, eyeCascade;
    faceCascade.load(faceCascadeName) ;
    eyeCascade.load(eyeCascadeName) ;
    
    
    Mat frameGray;
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    equalizeHist(frameGray, frameGray);
    vector<Rect> faces;
    vector<Point> centers;
    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for(int i=0;i<faces.size();i++){
        cout<<faces[i]<<endl;
        Mat faceROI = frameGray(faces[i]);
        vector<Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for(int j = 0; j < eyes.size(); j++){
            Point center( faces[i].x + eyes[j].x + int(eyes[j].width*0.5), faces[i].y + eyes[j].y + int(eyes[j].height*0.5) );
            centers.push_back(center);
        }
        if(centers.size()==2)
            break;
    }

    if(centers.size() == 2){
        Point leftPoint, rightPoint;
    
    //using haar cascade's coordinates obtained identify the left and right eyes
        if(centers[0].x < centers[1].x){
            leftPoint = centers[0];
            rightPoint = centers[1];
        }
        else{
            leftPoint = centers[1];
            rightPoint = centers[0];
        }
    
        w = 2.3 * (rightPoint.x - leftPoint.x);
        h = int(0.4 * w);
        x = leftPoint.x - 0.25*w;
        y = leftPoint.y - 0.5*h;
         Mat frameROI, eyeMaskSmall;
        Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
 
      //resizing the frames to the size of its eye region
        frameROI = frame(Rect(x,y,w,h));
        resize(eyeMask, eyeMaskSmall, Size(w,h));
        resize(opencvimg,opencvimg,Size(w,h));
        resize(bubble,bubble,Size(w,h));

        cvtColor(eyeMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);

        threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV);
        
        bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
        bitwise_and(eyeMaskSmall, eyeMaskSmall, maskedEye, grayMaskSmallThresh);
        
        bitwise_and(frameROI, frameROI, eyethreshold, grayMaskSmallThresh);
        
        bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
        
        bitwise_and(opencvimg,opencvimg,opencvframe,grayMaskSmallThresh);
        bitwise_and(bubble,bubble,bubbleframe,grayMaskSmallThresh);
    }
    
    createTrackbar("Reflection on the glass","Frame",&glassscalefactor,100,eyecv);
    createTrackbar("alpha of Bubbles","Frame",&bubblefactor,100,blenderbubble);
    createTrackbar("Transparency wrt Eye and glass","Frame",&scalefactor,100,eyeglass);
    
    waitKey(0);
    destroyAllWindows();
    return 0;
}
 
