/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.
 *
 * The omnidirectional camera in this module is denoted by the catadioptric
 * model. Please refer to Mei's paper for details of the camera model:
 *
 *      C. Mei and P. Rives, "Single view point omnidirectional camera
 *      calibration from planar grids", in ICRA 2007.
 *
 * The implementation of the calibration part is based on Li's calibration
 * toolbox:
 *
 *     B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
 *     Calibration Toolbox Using A Feature Descriptor-Based Calibration
 *     Pattern", in IROS 2013.
 */

#include "precomp.hpp"
#include "mymulticalib.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <filesystem>
#include <set>
namespace fs = std::filesystem;
namespace cv { namespace multicalib {

MyMultiCameraCalibration::MyMultiCameraCalibration(const std::vector<std::string> &a_cameraSerials, int cameraType, 
    int nCameras, const std::string& a_dataFolder,
    const std::string& a_cameraConfigFolder, const std::string a_DoubleSideConfig ,
    cv::Size frontPatternSize, cv::Size backPatternSize,
    float patternWidth, float patternHeight, int verbose, int showExtration, int nMiniMatches, int flags, TermCriteria criteria
):MultiCameraCalibration(cameraType, nCameras, a_dataFolder, patternWidth, patternHeight, verbose, showExtration, nMiniMatches, flags, criteria)
,cameraConfigFolder(a_cameraConfigFolder)
,cameraSerials(a_cameraSerials)
,dataFolder(a_dataFolder)
,_FrontPatternSize(frontPatternSize),
_BackPatternSize(backPatternSize)
{
    _cameraMatrix.clear(); 
    _distortCoeffs.clear();
    readcameraIntrinsics();
     if (a_DoubleSideConfig != ""){
        readDoubleSide(a_DoubleSideConfig);
        doublesideTransform2vec();
     }


    filesEachCameraFull.resize(nCameras);
    timestampFull.resize(nCameras);
    timestampAvailable.resize(nCameras);   
    timestampIsMulticamera.resize(nCameras);
}

void MyMultiCameraCalibration::readDoubleSide(const std::string a_DoubleSideConfig){
    if (a_DoubleSideConfig != ""){
        cv::FileStorage fs(a_DoubleSideConfig, cv::FileStorage::READ) ;
        fs ["transform"] >> doubleSideTransform;
    }
}
void MyMultiCameraCalibration::doublesideTransform2vec(){
    
    cv::Mat R;
    doubleSideTransform.rowRange(0, 3).colRange(0,3).copyTo(R);
    Rodrigues(R, doubleSideTransform_rvec);
    doubleSideTransform.rowRange(0, 3).col(3).copyTo(doubleSideTransform_tvec);

    doubleSideTransformInv = doubleSideTransform.inv();

    doubleSideTransformInv.rowRange(0, 3).colRange(0,3).copyTo(R);
    Rodrigues(R, doubleSideTransform_rvecInv);
    doubleSideTransformInv.rowRange(0, 3).col(3).copyTo(doubleSideTransform_tvecInv);
}
void MyMultiCameraCalibration::readcameraIntrinsics(){
    for(auto serial: cameraSerials){
        std::string filename = cameraConfigFolder +"/" + serial +".xml";
        cv::FileStorage fs(filename, cv::FileStorage::READ) ;
        cv::Mat Intrinsics, Distortion;
        fs["Intrinsics"] >> Intrinsics;
        fs["Distortion"] >> Distortion;
        Intrinsics.convertTo(Intrinsics, CV_32F);
        Distortion.convertTo(Distortion, CV_32F);

        _cameraMatrix.push_back(Intrinsics);
        _distortCoeffs.push_back(Distortion);

    }
}
cv::Mat transform2cvpoints(const cv::Mat & p){
    std::vector<cv::Point2d> points;
    for (int r = 0 ; r < p.rows; r++){
        auto x = p.at<double>(r,0); 
        auto y = p.at<double>(r,1); 
        cv::Point2d point = cv::Point2d (x, y);
        points.push_back(point);

    }
    cv::Mat matPoints = cv::Mat (points);
    return matPoints;
}
cv::Mat transform2cv3dpoints(const cv::Mat & p){
    std::vector<cv::Point3d> points;
    for (int r = 0 ; r < p.rows; r++){
        auto x = p.at<double>(r,0); 
        auto y = p.at<double>(r,1); 
        auto z = p.at<double>(r,2); 
        cv::Point3d point = cv::Point3d (x, y, z);
        points.push_back(point);

    }
    cv::Mat matPoints = cv::Mat (points);
    return matPoints;
}
Mat transposeObjpoints(const Mat &objectPoints){
    Mat objp = objectPoints.clone();
    int rows = 5;
    int cols = 8;

    for (int order0 = 0 ; order0 < objectPoints.rows ; order0 ++){
        int row0 = floor(order0 / cols);
        int col0 = order0 % cols;
        auto x = objectPoints.at<float>(order0, 0);
        auto y = objectPoints.at<float>(order0, 1);

        auto order1 = col0* rows + row0;
        printf("row0:%d, col0:%d,order1:%d, x:%f, y:%f\n",row0, col0, order1, x, y);
        objp.at<float>(order1,0) = x;
        objp.at<float>(order1,1) = y;
        objp.at<float>(order1,2) = 0;
    }
    for (int order0 = 0 ; order0 < objp.rows ; order0 ++){
        auto x = objp.at<float>(order0, 0);
        auto y = objp.at<float>(order0, 1);
        printf("x:%f, y:%f\n",x, y);
    }
    return objp;
}
void readCorners(const std::string &filename, cv::Mat& imagePoints, cv::Mat& objectPoints){
    cv::FileStorage fs = cv::FileStorage(filename, cv::FileStorage::READ);
    fs["corners"] >> imagePoints;
    // cv::Mat imagePoints1 = transform2cvpoints(imagePoints);
    // imagePoints = imagePoints1;
    fs["objects"] >> objectPoints;
    objectPoints.convertTo(objectPoints, CV_32F);    
    // objectPoints = transposeObjpoints(objectPoints);


    int tk = 0;
    // objectPoints = transform2cv3dpoints(objectPoints);

}

void MyMultiCameraCalibration::readCornersTo(const std::string &filepath, int cameraVertex, cv::Mat &imagePoints,
        cv::Mat & objectPoints){

        readCorners(filepath, imagePoints, objectPoints);

}


void MyMultiCameraCalibration::calcPatternPose(int cameraVertex, const cv::Mat &imagePoints,
        const cv::Mat & objectPoints,  cv::Mat &rvec, cv::Mat &tvec){

            cv::solvePnP(objectPoints, imagePoints, _cameraMatrix[cameraVertex], _distortCoeffs[cameraVertex], rvec, tvec);
            tvec.convertTo(tvec,CV_32F);
            assert(isValidPose(tvec));

        }

int MyMultiCameraCalibration::readTimestamps(const std::string &filepath, int cameraVertex){
        fs::path p = filepath;
        auto stem = p.stem();
        int timestamp = std::stoi(stem.string());

        return timestamp;
    }
void MyMultiCameraCalibration::storeReadedImp(const std::string &filepath, int cameraVertex,  
        int timestamp, cv::Mat &imagePoints,
        cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec){
        filesEachCameraFull[cameraVertex].push_back(filepath);
        timestampFull[cameraVertex].push_back(timestamp);
        timestampAvailable[cameraVertex].push_back(timestamp);   //
        _omEachCamera[cameraVertex].push_back(rvec);
        _tEachCamera[cameraVertex].push_back(tvec);
        imagePoints.convertTo(imagePoints, CV_32F);
        objectPoints.convertTo(objectPoints, CV_32F);
        _imagePointsForEachCamera[cameraVertex].push_back(imagePoints);
        _objectPointsForEachCamera[cameraVertex].push_back(objectPoints);
}
void MyMultiCameraCalibration::storeReaded(const std::string &filepath, int cameraVertex,  
        int timestamp, cv::Mat &imagePoints,
        cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec){
            if (imagePoints.rows == _FrontPatternSize.width * _FrontPatternSize.height){
                storeReadedImp(filepath, cameraVertex, timestamp, imagePoints, objectPoints, rvec, tvec);
            }
}
int MyMultiCameraCalibration::transformBackPattern(cv::Mat &objectPoints){
    if (isBackPattern(objectPoints)){

        // 背面的pattern 其objectpoints 坐标按doubleSideTransform 变换
        cv::Mat obTranspose = objectPoints.t();
        cv::Mat ones = Mat::ones(obTranspose.rows, obTranspose.cols, CV_64F);
        cv::Mat array[] ={obTranspose, ones };
        cv::Mat homogeneousObj;
        cv::vconcat(array, 2, homogeneousObj);
        assert(doubleSideTransform.at<double>(3, 0) == 0);
        cv::Mat TransformedObjectPoints = doubleSideTransform * homogeneousObj;
        objectPoints = TransformedObjectPoints;
    }
}

int MyMultiCameraCalibration::transformBackPatternPose(const cv::Mat &objectPoints,cv::Mat &rvec, cv::Mat &tvec){
    if (isBackPattern(objectPoints)){

        cv::Mat dom3dom1, dom3dT1, dom3dom2, dom3dT2, dT3dom1, dT3dT1, dT3dom2, dT3dT2;
        cv::Mat rvec3, tvec3;
        compose_motion(rvec, tvec, doubleSideTransform_rvec, doubleSideTransform_tvec , rvec3, tvec3,  dom3dom1,
        dom3dT1, dom3dom2, dom3dT2, dT3dom1, dT3dT1, dT3dom2, dT3dT2);
        rvec = rvec3;
        tvec = tvec3;
    }
}

void MyMultiCameraCalibration::loadOneSerial(int cameraVertex, const std::string&serial, const std::string&parfolder){
    const std::string folder = parfolder + "/" + serial;
    const std::string pattern = folder  + "/" + "*.yaml";
    
    std::vector<std::string> result;
    cv::glob(pattern, result);

    int invalidCnt = 0;
    for (auto file : result){
        if (m_outliers.find(file) != m_outliers.end())
        {
            std::cout <<"outlier:" << file <<" skipped: " << std::endl;
            continue;
        }
        int timestamp = readTimestamps(file, cameraVertex);
        cv::Mat imagePoints;
        cv::Mat objectPoints;
        
        readCornersTo(file, cameraVertex, imagePoints, objectPoints);
        // transformBackPattern(objectPoints);
        cv::Mat rvec, tvec;
        calcPatternPose(cameraVertex, imagePoints, objectPoints, rvec, tvec);
        // transformBackPatternPose(objectPoints, rvec, tvec);
        tvec.convertTo(tvec,CV_32F);
        if (isValidPose(tvec)){
            storeReaded(file, cameraVertex, timestamp, imagePoints, objectPoints, rvec, tvec);
        }
        else{
            invalidCnt ++;
            std::cout << "invalid pattern :" << invalidCnt <<", " << file<< std::endl;
            assert(false);
        }
    }
}

bool MyMultiCameraCalibration::findTimStamp( int cameraVertex, int a_timestamp, int numPatternPoints){
    const std::vector<int> &ta = timestampAvailable[cameraVertex];
    for (int i = 0;i < ta.size(); i++){
        auto timestamp = ta[i];
        if (timestamp == a_timestamp ){
            return true;
        }
    }
    return false;
}

void MyMultiCameraCalibration::identifyMultiCameraTimestamps(){
    // std::set<int> checked;
    setOfTimestampIsMulticamera.clear();
    for (int camera = 0; camera < _nCamera; ++camera){
        timestampIsMulticamera[camera].clear();
        for (int i = 0; i < timestampAvailable[camera].size(); ++i){
            auto &timestamp = timestampAvailable[camera][i];
            auto &imagePoints = _imagePointsForEachCamera[camera][i];

            // if (checked.find(timestamp)!=checked.end()){
            //     continue;
            // }
            // checked.insert(timestamp);
            bool isFoundTimestamp = false;
            for(int camera2 = 0; camera2 < _nCamera; ++camera2){
                if (camera2 == camera){
                    continue;
                }
                isFoundTimestamp = findTimStamp(camera2, timestamp, imagePoints.rows);
                if (isFoundTimestamp)
                {
                    break;
                }
            }
            if (isFoundTimestamp){
                timestampIsMulticamera[camera].push_back(true);
                setOfTimestampIsMulticamera.insert(timestamp);
            }else{
                timestampIsMulticamera[camera].push_back(false);
            }
            assert(timestampIsMulticamera[camera].size() == i+1);
        }
    }
}
void MyMultiCameraCalibration::loadImages(const std::set<std::string> &outliers)
{
    if (outliers.size() > 0)
    {
        m_outliers = outliers;
    }
    for (int camera = 0; camera < _nCamera; ++camera)
    {
        loadOneSerial(camera, cameraSerials[camera], dataFolder);
    }
    identifyMultiCameraTimestamps();
    // calibrate each camera individually
    for (int camera = 0; camera < _nCamera; ++camera)
    {
        Mat idx;
        idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
        for (int i = 0; i < (int)idx.total(); ++i)
        {
            idx.at<int>(i) = i;
        }

        for (int i = 0; i < (int)_omEachCamera[camera].size(); ++i)
        {
			int cameraVertex, timestamp, photoVertex;
			cameraVertex = camera;
			timestamp = timestampAvailable[camera][idx.at<int>(i)];
            if (setOfTimestampIsMulticamera.find(timestamp) == setOfTimestampIsMulticamera.end()){
                continue;
            }
			photoVertex = this->getPhotoVertex(timestamp);

			if (_omEachCamera[camera][i].type()!=CV_32F)
			{
				_omEachCamera[camera][i].convertTo(_omEachCamera[camera][i], CV_32F);
			}
			if (_tEachCamera[camera][i].type()!=CV_32F)
			{
				_tEachCamera[camera][i].convertTo(_tEachCamera[camera][i], CV_32F);
			}

			Mat transform = Mat::eye(4, 4, CV_32F);
			Mat R, T;
			Rodrigues(_omEachCamera[camera][i], R);
			T = (_tEachCamera[camera][i]).reshape(1, 3);
			R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
			T.copyTo(transform.rowRange(0, 3).col(3));
            

            auto eg = edge(cameraVertex, photoVertex, idx.at<int>(i), transform);
            auto &imgPoints = _imagePointsForEachCamera[cameraVertex][i];
            if (isBackPattern(imgPoints)){
                eg.patternSide = BACK_PATTERN;
            }
			this->_edgeList.push_back(eg);
        }

    }
}
std::set<std::string>  MyMultiCameraCalibration::removeOutlier() {
    std::vector<edge> edges;
    std::set<std::string> outlierNames;
    int cnt = 0;
    for (auto eg : this->_edgeList) {
        if (eg.reprojecterror > 0.5){
            auto filePath = filesEachCameraFull[eg.cameraVertex][eg.photoIndex];
            std::cout << "outlier:" <<eg.reprojecterror << ":" << filePath << " removed"<<std::endl;
            auto ret = outlierNames.insert(filePath);
            cnt ++;
        }else{
            edges.push_back(eg);
        }
    }
    std::cout <<"Totally " << cnt << " outlier removed" << std::endl;
    this->_edgeList = edges;
    return outlierNames;
}

void MyMultiCameraCalibration::writeParameters2config(){
    enum {scalarCnt = 2,
         matCnt = 3
    };
    std::string scalarNames[] = {"depth_scale", "height"};
    std::string matNames[] = {"CameraMatrix","Intrinsics","Distortion"};
    float scalars[scalarCnt];
    cv::Mat mats[matCnt];
    int camIdx = 0;
    for(auto serial: cameraSerials){
        std::string filename = cameraConfigFolder +"/" + serial +".xml";
        cv::FileStorage fs(filename, cv::FileStorage::READ) ;
        for (int i = 0; i < scalarCnt; i++){
            fs[scalarNames[i]] >> scalars[i] ;
        }
        for (int i = 0; i < matCnt; i++){
            fs[matNames[i]] >> mats[i];
        }    
        fs.release();
        mats[0] = _vertexList[camIdx].pose;
        cv::FileStorage fs1(filename, cv::FileStorage::WRITE) ;
        for (int i = 0; i < scalarCnt; i++){
            fs1 << scalarNames[i] << scalars[i];
        }
        for (int i = 0; i < matCnt; i++){
            fs1 << matNames[i] << mats[i];
        }    
        camIdx ++;              
    }
}


void MyMultiCameraCalibration::writeParameters(const std::string& filename) {
    MultiCameraCalibration::writeParameters(filename);
    writeParameters2config();
}

bool MyMultiCameraCalibration::isBackPattern(const cv::Mat & imgPoints) {
    return imgPoints.rows == _BackPatternSize.width * _BackPatternSize.height;
}



void MyMultiCameraCalibration::computePhotoCameraJacobian(int patternSide, const Mat& RvecPhoto, const Mat& TvecPhoto, const Mat& RvecCamera,
    const Mat& TvecCamera, const Mat& RvecDoubleside, const Mat& TvecDoubleside, 
    Mat& Rvectran, Mat& Tvectran,
    const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
    const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianCamera, Mat& E)
{
    //cameraPose * photoPose * doubleSideTransform = backpose;
    //cameraPose * photoPose  = frontPose; in camera coordinates
    Mat dRvectran_dRvecPhotofront, dRvectran_dTvecPhotofront,
        dTvectran_dRvecPhotofront, dTvectran_dTvecPhotofront,

        dRvectran_dRvecDoubleside, dRvectran_dTvecDoubleside,
        dTvectran_dRvecDoubleside, dTvectran_dTvecDoubleside,

        dRvectran_dRvecCamera, dRvectran_dTvecCamera,
        dTvectran_dRvecCamera, dTvectran_dTvecCamera;

    Mat dRvecPhotofront_dRvecPhoto, dRvecPhotofront_dTvecPhoto, 
        dTvecPhotofront_dRvecPhoto, dTvecPhotofront_dTvecPhoto,

        dRvecPhotofront_dRvecCamera, dRvecPhotofront_dTvecCamera,
        dTvecPhotofront_dRvecCamera, dTvecPhotofront_dTvecCamera;

    Mat dRvectran_dRvecPhoto, dRvectran_dTvecPhoto,
        dTvectran_dRvecPhoto, dTvectran_dTvecPhoto;

    cv::Mat RvecPhotofront, TvecPhotofront;
    // Mat dRvectran_dRvecPhotofront, dRvectran_dTvecPhotofront,
    // dTvectran_dRvecPhotofront, dTvectran_dTvecPhotofront;

    compose_motion(RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, RvecPhotofront, TvecPhotofront,
        dRvecPhotofront_dRvecPhoto, dRvecPhotofront_dTvecPhoto, dRvecPhotofront_dRvecCamera, dRvecPhotofront_dTvecCamera,
        dTvecPhotofront_dRvecPhoto, dTvecPhotofront_dTvecPhoto, dTvecPhotofront_dRvecCamera, dTvecPhotofront_dTvecCamera);

    Mat Rvectran1, Tvectran1;
    if (patternSide == BACK_PATTERN){
        compose_motion(RvecDoubleside, TvecDoubleside, RvecPhotofront, TvecPhotofront, Rvectran1, Tvectran1,
            dRvectran_dRvecDoubleside, dRvectran_dTvecDoubleside, dRvectran_dRvecPhotofront, dRvectran_dTvecPhotofront, 
            dTvectran_dRvecDoubleside, dTvectran_dTvecDoubleside, dTvectran_dRvecPhotofront, dTvectran_dTvecPhotofront);


        dRvectran_dRvecPhoto =  dRvectran_dRvecPhotofront * dRvecPhotofront_dRvecPhoto ;
        dRvectran_dTvecPhoto =  dRvectran_dTvecPhotofront * dTvecPhotofront_dTvecPhoto ;
        dTvectran_dRvecPhoto =  dTvectran_dRvecPhotofront * dRvecPhotofront_dRvecPhoto ;
        dTvectran_dTvecPhoto =  dTvectran_dTvecPhotofront * dTvecPhotofront_dTvecPhoto ;

        dRvectran_dRvecCamera =  dRvectran_dRvecPhotofront * dRvecPhotofront_dRvecCamera;
        dRvectran_dTvecCamera =  dRvectran_dTvecPhotofront * dTvecPhotofront_dTvecCamera;
        dTvectran_dRvecCamera =  dTvectran_dRvecPhotofront * dRvecPhotofront_dRvecCamera;
        dTvectran_dTvecCamera =  dTvectran_dTvecPhotofront * dTvecPhotofront_dTvecCamera;           

    }else{
 
        int rows = RvecPhotofront.total();

        dRvectran_dRvecPhoto =  dRvecPhotofront_dRvecPhoto;
        dRvectran_dTvecPhoto =  dRvecPhotofront_dTvecPhoto;
        dTvectran_dRvecPhoto =  dTvecPhotofront_dRvecPhoto;
        dTvectran_dTvecPhoto =  dTvecPhotofront_dTvecPhoto;  

        dRvectran_dRvecCamera =  dRvecPhotofront_dRvecCamera;
        dRvectran_dTvecCamera =  dRvecPhotofront_dTvecCamera;
        dTvectran_dRvecCamera =  dTvecPhotofront_dRvecCamera;
        dTvectran_dTvecCamera =  dTvecPhotofront_dTvecCamera;   

        Rvectran1 = RvecPhotofront;
        Tvectran1 = TvecPhotofront;
    }
    // std::cout <<"TvecCamera" << TvecCamera << std::endl;
    // std::cout <<"TvecPhoto" << TvecPhoto << std::endl;
  
    // std::cout <<"TvecPhotofront" << TvecPhotofront << std::endl;
    // std::cout <<"TvecCamera" << TvecCamera << std::endl;
    // std::cout <<"Tvectran" << Tvectran << std::endl;
    // std::cout <<"Tvectran1" << Tvectran1 << std::endl;
    


    if (Rvectran1.depth() == CV_64F)
    {
        Rvectran1.convertTo(Rvectran1, CV_32F);
    }
    if (Tvectran1.depth() == CV_64F)
    {
        Tvectran1.convertTo(Tvectran1, CV_32F);
    }



    float xif = 0.0f;
    if (_camType == OMNIDIRECTIONAL)
    {
        xif= xi.at<float>(0);
    }

    Mat imagePoints2, jacobian, dx_dRvecCamera, dx_dTvecCamera, dx_dRvecPhoto, dx_dTvecPhoto, E1;
    if (_camType == PINHOLE)
    {
        cv::projectPoints(objectPoints, Rvectran1, Tvectran1, K, distort, imagePoints2, jacobian);
        imagePoints2 = transform2colomn2(imagePoints2);
        assert (IsvalidImagePoints(imagePoints2));
    }
    //else if (_camType == FISHEYE)
    //{
    //    cv::fisheye::projectPoints(objectPoints, imagePoints2, Rvectran, Tvectran, K, distort, 0, jacobian);
    //}
    else if (_camType == OMNIDIRECTIONAL)
    {
        cv::omnidir::projectPoints(objectPoints, imagePoints2, Rvectran1, Tvectran1, K, xif, distort, jacobian);
    }
    if (objectPoints.depth() == CV_32F)
    {
        Mat(imagePoints - imagePoints2).convertTo(E1, CV_64FC2);
    }
    else
    {
        E1 = imagePoints - imagePoints2;
    }
    E = E1.reshape(1, (int)imagePoints.rows*2);
    std::cout << "norm(E)" << norm(E, NORM_INF) << std::endl;
    dx_dRvecCamera = jacobian.colRange(0, 3) * dRvectran_dRvecCamera + jacobian.colRange(3, 6) * dTvectran_dRvecCamera;
    dx_dTvecCamera = jacobian.colRange(0, 3) * dRvectran_dTvecCamera + jacobian.colRange(3, 6) * dTvectran_dTvecCamera;
    if (false && patternSide == BACK_PATTERN){
        std::cout<< "jacobian.colRange(0, 3):"<< jacobian.colRange(0, 3) << std::endl;
        std::cout<< "jacobian.colRange(3, 6):"<< jacobian.colRange(3, 6) << std::endl;
        std::cout<<"dRvectran_dRvecCamera:" << dRvectran_dRvecCamera<< std::endl;
        std::cout<<"dTvectran_dRvecCamera:" << dTvectran_dRvecCamera<< std::endl;
        std::cout<<"dRvectran_dTvecCamera:" << dRvectran_dTvecCamera<< std::endl;
        std::cout<<"dTvectran_dTvecCamera:" << dTvectran_dTvecCamera<< std::endl;
        std::cout<<"dx_dTvecCamera:" << dx_dTvecCamera<< std::endl;
        std::cout<<"dx_dRvecCamera:" << dx_dRvecCamera<< std::endl;
        int kkk=1;
    }


    dx_dRvecPhoto = jacobian.colRange(0, 3) * dRvectran_dRvecPhoto + jacobian.colRange(3, 6) * dTvectran_dRvecPhoto;
    dx_dTvecPhoto = jacobian.colRange(0, 3) * dRvectran_dTvecPhoto + jacobian.colRange(3, 6) * dTvectran_dTvecPhoto;

    jacobianCamera = cv::Mat(dx_dRvecCamera.rows, 6, CV_64F);
    jacobianPhoto = cv::Mat(dx_dRvecPhoto.rows, 6, CV_64F);

    dx_dRvecCamera.copyTo(jacobianCamera.colRange(0, 3));
    dx_dTvecCamera.copyTo(jacobianCamera.colRange(3, 6));
    dx_dRvecPhoto.copyTo(jacobianPhoto.colRange(0, 3));
    dx_dTvecPhoto.copyTo(jacobianPhoto.colRange(3, 6));

}
void MyMultiCameraCalibration::initialize() {

    G = buildGraph();
    // traverse the graph
    // std::vector<int> pre, order;
    graphTraverse(G, 0, order, pre);

    for (int i = 0; i < _nCamera; ++i)
    {
        if (pre[i] == INVALID)
        {
            std::cout << "camera" << i << "is not connected" << std::endl;
        }
    }

    for (int i = 1; i < (int)order.size(); ++i)
    {
        int vertexIdx = order[i];
        int preVertexIdx = pre[vertexIdx];
        Mat prePose = this->_vertexList[preVertexIdx].pose;
        int edgeIdx = G.at<int>(vertexIdx, preVertexIdx) - 1;
        edge &eg = this->_edgeList[edgeIdx];
        Mat transform = eg.transform.clone();
        if (eg.patternSide == BACK_PATTERN){
            //frontpose * doubleSideTransform = backpose
            //frontpose = backpose * doubleSideTransform.inv();            
            std::cout << "backpose" << transform<< std::endl;
            transform = transform * doubleSideTransformInv;
            std::cout << "frontPose" << transform<< std::endl;
            std::cout << "world frontPose" << prePose.inv() * transform<< std::endl;
            // std::cout << "cameraBackPose" << camerasPose[1] << std::endl;
        }
        if (vertexIdx < _nCamera)
        {
            this->_vertexList[vertexIdx].pose = transform * prePose.inv();
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
			if (_verbose)
			{
				std::cout << "initial pose for camera " << vertexIdx << " is " << std::endl;
				std::cout << this->_vertexList[vertexIdx].pose << std::endl;
			}
        }
        else
        {
            //cameraPose * photoPose =  frontpatternPoseInCamera;
            //cameraPose * photoPose =  frontpatternPoseInCamera = backpose * doubleSideTransformInv();
            //cameraPose * photoPose * doubleSideTransform = backpose;
            this->_vertexList[vertexIdx].pose = prePose.inv() * transform;
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
        }
    }   

}
void MyMultiCameraCalibration::computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE, Mat&deltaX)
{
    int nParam = (int)extrinsicParams.total();
    int nEdge = (int)_edgeList.size();
    std::vector<int> pointsLocation(nEdge+1, 0);
    
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int nPoints = (int)_objectPointsForEachCamera[_edgeList[edgeIdx].cameraVertex][_edgeList[edgeIdx].photoIndex].rows;
        pointsLocation[edgeIdx+1] = pointsLocation[edgeIdx] + nPoints*2;
    }

    JTJ_inv = Mat(nParam, nParam, CV_64F);
    JTE = Mat(nParam, 1, CV_64F);

    Mat J = Mat::zeros(pointsLocation[nEdge], nParam, CV_64F);
    Mat E = Mat::zeros(pointsLocation[nEdge], 1, CV_64F);
    Mat RvecDoubleSide = doubleSideTransform_rvec;
    Mat TvecDoubleSide = doubleSideTransform_tvec;

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int photoVertex = _edgeList[edgeIdx].photoVertex;
        int photoIndex = _edgeList[edgeIdx].photoIndex;
        int cameraVertex = _edgeList[edgeIdx].cameraVertex;

        Mat objectPoints = _objectPointsForEachCamera[cameraVertex][photoIndex];
        Mat imagePoints = _imagePointsForEachCamera[cameraVertex][photoIndex];

        Mat Rvectran, Tvectran;
        edge &eg = _edgeList[edgeIdx];
        Mat transform = eg.transform;
        // if (eg.patternSide == BACK_PATTERN){
        //     transform = transform * doubleSideTransformInv;
        // }
        Mat R = transform.rowRange(0, 3).colRange(0, 3);
        Tvectran = transform.rowRange(0, 3).col(3);
        cv::Rodrigues(R, Rvectran);
        assert(isValidPose(Tvectran));

        //为什么是Photovertex,而不是photoIndex


        int paraRow = photoVertex-1;
        Mat RvecPhoto = extrinsicParams.colRange(paraRow*6, paraRow*6 + 3);
        Mat TvecPhoto = extrinsicParams.colRange(paraRow*6 + 3, paraRow*6 + 6);
        assert(isValidPose(TvecPhoto));
        Mat RvecCamera, TvecCamera;
        if (cameraVertex > 0)
        {
            RvecCamera = extrinsicParams.colRange((cameraVertex - 1)*6, (cameraVertex - 1)*6 + 3);
            TvecCamera = extrinsicParams.colRange((cameraVertex - 1)*6 + 3, (cameraVertex - 1)*6 + 6);
        }
        else
        {
            RvecCamera = Mat::zeros(3, 1, CV_32F);// 第一个相机姿态无旋转平移
            TvecCamera = Mat::zeros(3, 1, CV_32F);
        }
        const float epsilon = 0.001;
        cv::Mat TvecDoubleSideEpsilon = TvecDoubleSide.clone();
        TvecDoubleSideEpsilon.at<float>(0) += epsilon;
        // std::cout << "TvecDoubleSideEpsilon:" << TvecDoubleSideEpsilon << std::endl;
        // std::cout << "TvecDoubleSide:" << TvecDoubleSide << std::endl;

        cv::Mat RvecDoubleSideEpsilon = RvecDoubleSide.clone();
        const int col = 0;
        RvecDoubleSideEpsilon.at<float>(col) += epsilon;
       

        Mat jacobianPhoto, jacobianCamera, error, error1, error2;
        // computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
        //     RvecDoubleSide, TvecDoubleSideEpsilon, Rvectran, Tvectran, 
        //     objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
        //     this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error2);        

        // computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
        //     RvecDoubleSideEpsilon, TvecDoubleSide, Rvectran, Tvectran, 
        //     objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
        //     this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error1);  

        computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
            RvecDoubleSide, TvecDoubleSide, Rvectran, Tvectran, 
            objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
            this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error);

        // if (eg.patternSide == BACK_PATTERN){
        //     cv::Mat errordiff = error1 - error;
        //     cv::Mat differenceRvec =errordiff / epsilon;
        //     cv::Mat differenceTvec = (error2 - error) / epsilon;
        //     cv::Mat derivative = jacobianCamera.col(col);
        //     // cv::Mat accuracy = difference - derivative;
        //     // std::cout << "RvecDoubleSideEpsilon:" << RvecDoubleSideEpsilon << std::endl;
        //     // std::cout << "RvecDoubleSide:" << RvecDoubleSide << std::endl;             
        //     // std::cout << "jacobianCamera" << jacobianCamera<< std::endl;
        //     std::cout << "differenceRvec" << differenceRvec.t() << std::endl;
        //     std::cout << "derivativeRvec" << derivative.t() << std::endl;
        //     std::cout << "differenceTvec" << differenceTvec.t() << std::endl;
        //     std::cout << "derivativeTvec" << jacobianCamera.col(3).t() << std::endl;
            
        //     // std::cout << "accuracy" << accuracy.t() << std::endl;
        //     int kk1 = 1;
        // }

        // jacobianCamera.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
        //         colRange((cameraVertex-1)*6, cameraVertex*6));
        // jacobianPhoto.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
        //     colRange((photoVertex-1)*6, photoVertex*6));
        // error.copyTo(E.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]));
        int rowBegin = pointsLocation[edgeIdx];
        int rowEnd = pointsLocation[edgeIdx+1];
        if (cameraVertex > 0)
        {
            jacobianCamera.copyTo(J.rowRange(rowBegin, rowEnd).
                colRange((cameraVertex-1)*6, cameraVertex*6));
        }
        
        jacobianPhoto.copyTo(J.rowRange(rowBegin, rowEnd).
            colRange(paraRow*6, (paraRow + 1)*6));
        error.copyTo(E.rowRange(rowBegin, rowEnd));
        if (false && eg.patternSide == BACK_PATTERN){
            std::cout << "jacobianCamera:"<<jacobianCamera<< std::endl;
            std::cout << "jacobianPhoto:"<<jacobianPhoto << std::endl;
        }
    }
    // std::cout << "J:"<<J << std::endl;
    //std::cout << J.t() * J << std::endl;
    
    // cv::Mat Q,R;
    // std::cout << "QRdecompose begin ..."<<  std::endl;
    // QRdecompose(J, Q, R);
    // auto end = std::chrono::steady_clock::now();
    // auto elpased = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "QRdecompose taks milliseconds:"<< elpased << std::endl;
    // auto deltX = -R.inv() * Q.t() * E;
    cv::Mat JTJ = (J.t() * J );
    JTE = J.t() * E;
    auto start =std::chrono::steady_clock::now();
    auto x = conjungate(JTJ, JTE);
    std::cout<<"x:" << x.t() << std::endl;
    // std::cout<<"E:" << E.t() << std::endl;
    deltaX = x;
    auto end1 = std::chrono::steady_clock::now();
    auto elpased1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count();   
    // std::cout << "inv taks milliseconds:"<< elpased1 << std::endl; 
    // JTJ_inv = JTJ.inv();
    // auto end = std::chrono::steady_clock::now();
    // auto elpased = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "inv taks milliseconds:"<< elpased << std::endl;
    

}

double MyMultiCameraCalibration::computeProjectError(Mat& parameters)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.total() == (nVertex-1) * 6 && parameters.depth() == CV_32F);
    int nEdge = (int)_edgeList.size();

    // recompute the transform between photos and cameras

    std::vector<edge> edgeList = this->_edgeList;
    std::vector<Vec3f> RvecVertex, TvecVertex;
    vector2parameters(parameters, RvecVertex, TvecVertex);
    std::cout << "TvecVertex[0]" << TvecVertex[0] << std::endl;
    float totalError = 0;
    int totalNPoints = 0;
    std::vector<float> errorsVector;
    std::vector<float> errorsVectorPerImage;
    Mat photoPose , doubleSideTransform1 , transform;
    photoPose = Mat::eye(4, 4, CV_32F);
    doubleSideTransform1 = doubleSideTransform;
    transform = Mat::eye(4, 4, CV_32F);
    Mat Tdoubleside = doubleSideTransform_tvec;
    Mat Rdoubleside = doubleSideTransform_rvec;

    //cameraPose * photoPose * doubleSideTransform = backpose
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat RPhoto, RCamera, TPhoto, TCamera;
        auto &edg = edgeList[edgeIdx];
        int cameraVertex = edg.cameraVertex;
        int photoVertex = edg.photoVertex;
        int PhotoIndex = edg.photoIndex;
        TPhoto = Mat(TvecVertex[photoVertex - 1]).reshape(1, 3);

        //edgeList[edgeIdx].transform = Mat::ones(4, 4, CV_32F);
        transform = Mat::eye(4, 4, CV_32F);
        cv::Rodrigues(RvecVertex[photoVertex-1], RPhoto);
        if (cameraVertex == 0)
        {
            RPhoto.copyTo(transform.rowRange(0, 3).colRange(0, 3));
            TPhoto.copyTo(transform.rowRange(0, 3).col(3));
        }
        else
        {
            TCamera = Mat(TvecVertex[cameraVertex - 1]).reshape(1, 3);
            cv::Rodrigues(RvecVertex[cameraVertex - 1], RCamera);
            Mat(RCamera*RPhoto).copyTo(transform.rowRange(0, 3).colRange(0, 3));
            Mat(RCamera * TPhoto + TCamera).copyTo(transform.rowRange(0, 3).col(3));
        }


        transform.copyTo(edg.transform);
        Mat rvec, tvec;
        cv::Rodrigues(transform.rowRange(0, 3).colRange(0, 3), rvec);
        transform.rowRange(0, 3).col(3).copyTo(tvec);

        Mat objectPoints, imagePoints, proImagePoints;
        objectPoints = this->_objectPointsForEachCamera[cameraVertex][PhotoIndex];
        imagePoints = this->_imagePointsForEachCamera[cameraVertex][PhotoIndex];

        if (this->_camType == PINHOLE)
        {
            cv::projectPoints(objectPoints, rvec, tvec, _cameraMatrix[cameraVertex], _distortCoeffs[cameraVertex],
                proImagePoints);
            proImagePoints = transform2colomn2(proImagePoints);
            int k = 0;
        }
        //else if (this->_camType == FISHEYE)
        //{
        //    cv::fisheye::projectPoints(objectPoints, proImagePoints, rvec, tvec, _cameraMatrix[cameraVertex],
        //        _distortCoeffs[cameraVertex]);
        //}
        else if (this->_camType == OMNIDIRECTIONAL)
        {
            float xi = _xi[cameraVertex].at<float>(0);

            cv::omnidir::projectPoints(objectPoints, proImagePoints, rvec, tvec, _cameraMatrix[cameraVertex],
                xi, _distortCoeffs[cameraVertex]);
        }
        Mat error = imagePoints - proImagePoints;
        Vec2f* ptr_err = error.ptr<Vec2f>();
        Vec2f* pimagePoints = imagePoints.ptr<Vec2f>();
        Vec2f* pproImagePoints = proImagePoints.ptr<Vec2f>();
        float errorPerImage = 0;
        for (int i = 0; i < (int)error.rows; ++i)
        {
            float ferror = sqrt(ptr_err[i][0]*ptr_err[i][0] + ptr_err[i][1]*ptr_err[i][1]);
            errorsVector.push_back(ferror);
            
            errorPerImage += ferror;
            if (ferror > 1000){
                isValidPose(tvec);
            }
        }
        edg.reprojecterror = errorPerImage/error.rows;
        _edgeList[edgeIdx].reprojecterror = errorPerImage/error.rows;
        errorsVectorPerImage.push_back(errorPerImage);
        totalError += errorPerImage;
        totalNPoints += (int)error.total();
    }
    std::vector<edge> sortedEdgelist;
    sortedgelist(edgeList, sortedEdgelist);
    printedgelist(sortedEdgelist);

    double meanReProjError = totalError / totalNPoints;
    _error = meanReProjError;
    std::cout << "totalError:"<< totalError << std::endl;
     std::cout << "totalNPoints:"<< totalNPoints << std::endl;
    std::cout << "meanReProjError:"<< meanReProjError << std::endl;

    float var = 0;
    for(int n = 0; n < errorsVector.size(); n++ )
    {
        var += (errorsVector[n] - meanReProjError) * (errorsVector[n] - meanReProjError);
    }

    var /= errorsVector.size();
    auto sd = sqrt(var); 
    std::cout << "standard deviation of ReProjError:"<< sd << std::endl;
    return meanReProjError;
}
}} // namespace multicalib, cv
