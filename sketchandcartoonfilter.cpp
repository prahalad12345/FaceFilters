#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace std;
using namespace cv;

/*
    implementing cartoon filter and sketch filter using gradient 
    Cartoon filter:
    the gray scale image is differenciated to find get an outline border
    gradient provides the change from one pixel to the next.
    obtain the inverse of the diffentiated image

    the bgr image is gaussian blurred to get an even toned image
    
    merge the two images to obtain the final cartoonified image

    Sketch image:
    Involves the differentiation part mentioned above
*/
Mat cartoonification(Mat image){
    
    Mat cartoonImage;                              
    Mat sketch;
    cvtColor(image,sketch,COLOR_BGR2GRAY);
    GaussianBlur(sketch,sketch,Size(3,3),0);
    Laplacian(sketch,sketch,CV_8U,1,13);
    sketch=255-sketch;
    threshold(sketch,sketch,195,255,THRESH_BINARY);

    vector<Mat> sketches(3);
    sketches[0]=sketch;
    sketches[1]=sketch;
    sketches[2]=sketch;
    Mat finalsketch;
    merge(sketches,finalsketch);
    bitwise_and(finalsketch,image,cartoonImage);
    
    return cartoonImage;

}


Mat pencilSketch(Mat image){
    
    Mat pencilSketchImage;
    Mat sketch;
    cvtColor(image,sketch,COLOR_BGR2GRAY);
    GaussianBlur(sketch,sketch,Size(3,3),0);
    Laplacian(sketch,sketch,CV_8U,1,13);
    /// YOUR CODE HERE
    sketch=255-sketch;
    threshold(sketch,pencilSketchImage,195,255,THRESH_BINARY);
    
    return pencilSketchImage;
}


int main(int argc, char *argv[]){
    Mat image=imread("./images/2.jpg");
    Mat pencilimage=pencilSketch(image);
    imwrite("./images/pencilsketch.jpg",pencilimage);
    Mat cartoonimage=cartoonify(image);
    imwrite("./images/cartoonify.jpg",cartoonimage);
}
