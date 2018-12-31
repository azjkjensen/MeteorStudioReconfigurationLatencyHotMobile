#include <jni.h>
#include <com_msrs_opencv_android_NativeCallMethods.h>
#include <android/log.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <android/native_window.h>
#include <android/native_window_jni.h>


#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "sampleMSRS", __VA_ARGS__)

using namespace cv;
using namespace std;

vector<KeyPoint> keypointsReference;
Mat descriptorsReference;
vector<Point3f>  p3d;
int numFeautresReference = 500;
int numFeaturesDest = 500;


JNIEXPORT void JNICALL
        Java_com_msrs_pose_1estimation_NativeCallMethods_poseEstimateNative(JNIEnv *env, jclass type,
                                                                          jint srcWidth, jint srcHeight,
                                                                          jobject srcBuffer, jobject dstSurface,
                                                                          jboolean colorFlag, jbyteArray byteArray)
{

    //acquiring the image buffer
    uint8_t *srcLumaPtr = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(srcBuffer));

    int dstWidth;
    int dstHeight;

    cv::Mat mYuv(srcHeight + srcHeight / 2, srcWidth, CV_8UC1, srcLumaPtr);


    ANativeWindow *win = ANativeWindow_fromSurface(env, dstSurface);
    ANativeWindow_acquire(win);

    ANativeWindow_Buffer buf;
    dstWidth = srcHeight;
    dstHeight = srcWidth;


    ANativeWindow_setBuffersGeometry(win, dstWidth, dstHeight, 0 );

    //acquiring a lock on the SurfaceView to render the processed image
    if (int32_t err = ANativeWindow_lock(win, &buf, NULL)) {
        LOGE("ANativeWindow_lock failed with error code %d\n", err);
        ANativeWindow_unlockAndPost(win);
        ANativeWindow_release(win);
        return ;
    }

    uint8_t *dstLumaPtr = reinterpret_cast<uint8_t *>(buf.bits);

    Mat dstRgba(dstHeight, buf.stride, CV_8UC4,
                dstLumaPtr);
    Mat srcRgba(srcHeight, srcWidth, CV_8UC4);
    Mat flipRgba(dstHeight, dstWidth, CV_8UC4);
    Mat colorRgba(dstHeight, dstWidth, CV_8UC4);


    // convert YUV -> RGBA
    cv::cvtColor(mYuv, colorRgba, CV_YUV2RGBA_NV21);
    if(colorFlag)
        cv::cvtColor(mYuv, srcRgba, CV_YUV2RGBA_NV21);

    else {
        cv::cvtColor(mYuv,srcRgba,CV_YUV420p2GRAY);
//        cv::cvtColor(srcRgba,srcRgba,CV_GRAY2RGB);

    }
    // Rotate 90 degree, Comment/Uncomment this depending on the phone
//    coloRgba = srcRgba;
    cv::transpose(srcRgba, flipRgba);
    cv::flip(flipRgba, flipRgba,1);// negative parameter indicates flipping on both axes. 0- flipping on y-axis and 1- splitting on x-axis.
    //this is just for showing the color image instead of the black and the white one
    cv::transpose(colorRgba, colorRgba);
    cv::flip(colorRgba, colorRgba,1);// negative parameter indicates flipping on both axes. 0- flipping on y-axis and 1- splitting on x-axis.

    const double cx = flipRgba.cols/2;
    const double cy = flipRgba.rows/2;
    const double fx = 1.73*cx;
    const double fy=1.73*cy;
    //adding our homography code here
    //starting with extracting feature points from the destination image
    std::vector<cv::KeyPoint> keypointsDest;
    cv::Mat descriptorsDest;

    //extracting keypoints from the incoming image frame
    Ptr<FeatureDetector> detector = ORB::create(numFeaturesDest,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    detector->detect(flipRgba, keypointsDest);
    detector->compute(flipRgba, keypointsDest, descriptorsDest);

//    cv::cvtColor(flipRgba,flipRgba,CV_RGBA2RGB);
//    drawKeypoints(flipRgba,keypointsDest,flipRgba,Scalar(255,0,0));

    //extracting keypoints done
    cv::Mat distances;
    int k=2;
    std::vector< std::vector<cv::DMatch> > knnMatches;

    if(descriptorsReference.type()==CV_8U)
    { //if orb


        //using a FLANN based matcher to match keypoints from Reference Image and Input Image
        cv::FlannBasedMatcher  matcher(new flann::LshIndexParams(20,10,2));
//        descriptorsDest.convertTo(descriptorsDest,CV_32F);
//        descriptorsReference.convertTo(descriptorsReference,CV_32F);

        if(descriptorsDest.rows>20) {
            matcher.knnMatch(
                    descriptorsDest,
                    descriptorsReference,
                    knnMatches,
                    k);
        }

    }
    else
    {
        //if SIFT
        cv::FlannBasedMatcher  matcher;

        if(descriptorsDest.rows>20) {
            matcher.knnMatch(
                    descriptorsDest,
                    descriptorsReference,
                    knnMatches,
                    k);
        }
    }

    if(distances.type() == CV_32S)
    {
        cv::Mat temp;
        distances.convertTo(temp, CV_32F);
        distances = temp;
    }

    std::vector<cv::KeyPoint> kp2;
    std::vector<cv::DMatch> goodMatches;

    //pass the matches through a ratio test to determine the best matches
    for (size_t i = 0; i<knnMatches.size(); i++) {
        const cv::DMatch& match1 = knnMatches[i][0];
        const cv::DMatch& match2 = knnMatches[i][1];


        if (match1.distance < 0.8*match2.distance)
            goodMatches.push_back(match1);
    }

    /* uncomment to see matched keypoints drawn on the frame
    for (unsigned int i = 0; i<matches.size(); i++) {
        int i2 = matches[i].queryIdx;
        kp2.push_back(keypointsDest[i2]);
    }

    drawKeypoints(flipRgba, kp2, flipRgba, cv::Scalar(0, 255, 255), cv::DrawMatchesFlags::DEFAULT);
    */

    cv::cvtColor(flipRgba,flipRgba,CV_GRAY2RGBA);

    if (goodMatches.size() < 4) {
        //findHomography needs a minimum of 4 points.
    }
    else {
        std::vector<cv::Point2f> pts1(goodMatches.size());
        std::vector<cv::Point2f> pts2(goodMatches.size());
        for (size_t i = 0; i < goodMatches.size(); i++) {
            pts1[i] = keypointsReference[goodMatches[i].trainIdx].pt;
            pts2[i] = keypointsDest[goodMatches[i].queryIdx].pt;
        }


        std::vector<unsigned char> inliersMask(pts1.size());
        cv::Mat homography = cv::findHomography(pts1, pts2, cv::FM_RANSAC, 5, inliersMask);

        std::vector<cv::DMatch> inliers;
        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i])
                inliers.push_back(goodMatches[i]);
        }

        // If too few points to compute pose, skip this image.
        if (inliers.size() >5) {

            int len = env->GetArrayLength(byteArray);
            unsigned char* buf = new unsigned char[len];
            env->GetByteArrayRegion (byteArray, 0, len, reinterpret_cast<jbyte*>(buf));
//            p3d = (std::vector<cv::Point3_>)buf;
            Point3f * bf = reinterpret_cast<Point3f *>(buf);
            p3d = *(new std::vector<cv::Point3f>(bf, bf + len));
//            Mat((int)vec.size(), 1, traits::Type<_Tp>::value, (uchar*)&vec[0]);

            std::vector<cv::Point2f> p2D;
            std::vector<cv::Point3f> p3D;
            for (unsigned int i = 0; i < inliers.size(); i++) {

                int i1 = inliers[i].trainIdx;
                p3D.push_back(p3d[i1]);
                int i2 = inliers[i].queryIdx;
                p2D.push_back(keypointsDest[i2].pt);
            }

            double data[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};

            //make the camera intrinsic parameters matrix
            cv::Mat K = cv::Mat(3, 3, CV_64F, data);

            cv::Mat rotVec, transVec;
            bool foundPose = cv::solvePnP(p3D, p2D, K, cv::Mat::zeros(5, 1, CV_64F), rotVec, transVec);

            if (foundPose) {
                std::vector<cv::Point3d> pointsAxes;
                std::vector<cv::Point3d> pointAxesForCube;
                std::vector<cv::Point2d> pointsImage;
                std::vector<cv::Point2d> pointsImagesRoof;


                int axisLength = 5;

                //the following commented code draws the x,y,z axis
//                pointsAxes.push_back(cv::Point3d(0, 0, 0));
//                pointsAxes.push_back(cv::Point3d(axisLength, 0, 0));
//                pointsAxes.push_back(cv::Point3d(0, axisLength, 0));
//                pointsAxes.push_back(cv::Point3d(0, 0, axisLength));

//                cv::projectPoints(pointsAxes, rotVec, transVec, K, cv::Mat::zeros(5, 1, CV_64F),
//                                  pointsImage);
//
//                line(flipRgba, pointsImage[0], pointsImage[1], cv::Scalar(0, 0, 255), 2);
//                line(flipRgba, pointsImage[0], pointsImage[2], cv::Scalar(0, 255, 0), 2);
//                line(flipRgba, pointsImage[0], pointsImage[3], cv::Scalar(255, 0, 0), 2);

                //drawing the cube
                pointAxesForCube.push_back(cv::Point3d(0,0,0));
                pointAxesForCube.push_back(cv::Point3d(0,axisLength,0));
                pointAxesForCube.push_back(cv::Point3d(axisLength,axisLength,0));
                pointAxesForCube.push_back(cv::Point3d(axisLength,0,0));
                pointAxesForCube.push_back(cv::Point3d(0,0,-axisLength));
                pointAxesForCube.push_back(cv::Point3d(0,axisLength,-axisLength));
                pointAxesForCube.push_back(cv::Point3d(axisLength,axisLength,-axisLength));
                pointAxesForCube.push_back(cv::Point3d(axisLength,0,-axisLength));

                cv::projectPoints(pointAxesForCube, rotVec, transVec, K, cv::Mat::zeros(5, 1, CV_64F),
                                  pointsImage);

                vector<vector<Point> > pointsImageFloor(1);

                for(int i=0;i<4;i++){
                    pointsImageFloor[0].push_back(Point((int)pointsImage[i].x,(int)pointsImage[i].y));
                }
                for(int i=4;i<8;i++){
                    pointsImagesRoof.push_back(pointsImage[i]);
                }
                //drawing the floor of the box
                drawContours(colorRgba,pointsImageFloor, -1, (255, 0, 0), CV_FILLED,LINE_8);

                for(int i=0;i<4;i++){
                    line(colorRgba,pointsImage[i],pointsImage[i+4],Scalar(255,223,223),3);
                }
                line(colorRgba,pointsImage[0],pointsImage[1],Scalar(255,223,223),3);
                line(colorRgba,pointsImage[1],pointsImage[2],Scalar(255,223,223),3);
                line(colorRgba,pointsImage[2],pointsImage[3],Scalar(255,223,223),3);
                line(colorRgba,pointsImage[3],pointsImage[0],Scalar(255,223,223),3);
                line(colorRgba,pointsImage[4],pointsImage[5],Scalar(255,223,223),3);
                line(colorRgba,pointsImage[5],pointsImage[6],Scalar(255,233,233),3);
                line(colorRgba,pointsImage[6],pointsImage[7],Scalar(255,233,233),3);
                line(colorRgba,pointsImage[7],pointsImage[4],Scalar(255,233,233),3);


            }
        }
    }


    // copy to TextureView surface
    uchar *dbuf;
    uchar *sbuf;
    dbuf = dstRgba.data;
    sbuf = colorRgba.data;
    int i;
    for (i = 0; i < colorRgba.rows; i++) {
        dbuf = dstRgba.data + i * buf.stride * 4;
        memcpy(dbuf, sbuf, colorRgba.cols*4);
        sbuf += colorRgba.cols * 4;
    }


    ANativeWindow_unlockAndPost(win);
    ANativeWindow_release(win);

}

JNIEXPORT jbyteArray JNICALL
Java_com_msrs_pose_1estimation_NativeCallMethods_generateKeypointsReferenceNative(JNIEnv *env,
                                                                              jclass type) {
    int vSize = sizeof(std::vector<cv::Point3f>) + (sizeof(Point3f) * keypointsReference.size());
    char* bytes = new char[vSize]();
//    memcpy(bytes, (void*) p3d.data(), vSize);
//    env->ReleaseByteArrayElements()
    jbyteArray out = env->NewByteArray(vSize);
    env->SetByteArrayRegion(out, 0, vSize, (const jbyte*) keypointsReference.data());
//    jboolean isCopy;
//    jbyte *jbuf = env->GetCharArrayElements((jchar*)m.data, &isCopy);
    return out;
}

JNIEXPORT jbyteArray JNICALL
Java_com_msrs_pose_1estimation_NativeCallMethods_generateDescriptorsReferenceNative(JNIEnv *env,
                                                                              jclass type) {

    int vSize = descriptorsReference.total() * descriptorsReference.elemSize();
    char* bytes = new char[vSize]();
//    memcpy(bytes, (void*) p3d.data(), vSize);
//    env->ReleaseByteArrayElements()
    std::memcpy(bytes,descriptorsReference.data, vSize);
    jbyteArray out = env->NewByteArray(vSize);
    env->SetByteArrayRegion(out, 0, vSize, (const jbyte*) bytes);
//    jboolean isCopy;
//    jbyte *jbuf = env->GetCharArrayElements((jchar*)m.data, &isCopy);
    return out;
}

// Need to pass back keypointsReference, descriptorsReference, and p3d.
JNIEXPORT jbyteArray JNICALL
Java_com_msrs_pose_1estimation_NativeCallMethods_generateReferenceImageNative(JNIEnv *env,
                                                                             jclass type,
                                                                             jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);

    Mat referenceImage = imread(path);
//    Mat* matOut =(Mat*) matPtr;

    Ptr<FeatureDetector> detector = ORB::create(numFeautresReference,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);

    //extracting keypoints and descriptors once for the refernce image and storing them as global variables.
    detector->detect(referenceImage, keypointsReference);
    detector->compute(referenceImage, keypointsReference, descriptorsReference);

    const double heightAbove = 25.0;
    const double cx= 918.0;
    const double cy =637.0;
    const double fx = cx*1.73;
    const double fy = cy*1.73;

    for (int i = 0; i<keypointsReference.size(); i++) {
        float x = keypointsReference[i].pt.x;	// 2D location in image
        float y = keypointsReference[i].pt.y;
        float X = (heightAbove / fx)*(x - cx);
        float Y = (heightAbove / fy)*(y - cy);
        float Z = 0;
        p3d.push_back(cv::Point3f(X, Y, Z));
    }
    env->ReleaseStringUTFChars(path_, path);
//    *matOut = Mat(p3d, true);
//    Mat m = Mat(p3d, true);
//    m.data;
    int vSize = sizeof(std::vector<cv::Point3f>) + (sizeof(Point3f) * p3d.size());
    char* bytes = new char[vSize]();
//    memcpy(bytes, (void*) p3d.data(), vSize);
//    env->ReleaseByteArrayElements()
    jbyteArray p3d_out = env->NewByteArray(vSize);
    env->SetByteArrayRegion(p3d_out, 0, vSize, (const jbyte*) p3d.data());
//    jboolean isCopy;
//    jbyte *jbuf = env->GetCharArrayElements((jchar*)m.data, &isCopy);
    return p3d_out;
}

JNIEXPORT void JNICALL
Java_com_msrs_pose_1estimation_NativeCallMethods_setDescriptorsReferenceNative(JNIEnv *env,
                                                                               jclass type, jbyteArray dRef){
    int len = env->GetArrayLength(dRef);
    unsigned char* buf = new unsigned char[len];
    env->GetByteArrayRegion(dRef, 0, len, reinterpret_cast<jbyte*>(buf));

    descriptorsReference = Mat(500, 32, 0, buf, CV_AUTO_STEP);
//    descriptorsReference.data = buf;
}

JNIEXPORT void JNICALL
Java_com_msrs_pose_1estimation_NativeCallMethods_setKeypointsReferenceNative(JNIEnv *env,
                                                                             jclass type, jbyteArray kRef){
    int len = env->GetArrayLength(kRef);
    unsigned char* buf = new unsigned char[len];
    env->GetByteArrayRegion (kRef, 0, len, reinterpret_cast<jbyte*>(buf));
//            p3d = (std::vector<cv::Point3_>)buf;
    KeyPoint * bf = reinterpret_cast<KeyPoint *>(buf);
    keypointsReference = *(new std::vector<KeyPoint>(bf, bf + len));
}

