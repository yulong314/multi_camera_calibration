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

#ifndef __OPENCV_MYMULTICAMERACALIBRATION_HPP__
#define __OPENCV_MYMULTICAMERACALIBRATION_HPP__


#include <string>
#include <iostream>
#include "multicalib.hpp"
#include <set>
using namespace cv;

namespace cv { namespace multicalib {

//! @addtogroup ccalib
//! @{


/** @brief Class for multiple camera calibration that supports pinhole camera and omnidirection camera.
For omnidirectional camera model, please refer to omnidir.hpp in ccalib module.
It first calibrate each camera individually, then a bundle adjustment like optimization is applied to
refine extrinsic parameters. So far, it only support "random" pattern for calibration,
see randomPattern.hpp in ccalib module for details.
Images that are used should be named by "cameraIdx-timestamp.*", several images with the same timestamp
means that they are the same pattern that are photographed. cameraIdx should start from 0.

For more details, please refer to paper
    B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
    Calibration Toolbox Using A Feature Descriptor-Based Calibration
    Pattern", in IROS 2013.
*/

class CV_EXPORTS MyMultiCameraCalibration: public MultiCameraCalibration
{
public:
 
    /* @brief Constructor
    @param cameraType camera type, PINHOLE or OMNIDIRECTIONAL
    @param nCameras number of cameras
    @fileName filename of string list that are used for calibration, the file is generated
    by imagelist_creator from OpenCv samples. The first one in the list is the pattern filename.
    @patternWidth the physical width of pattern, in user defined unit.
    @patternHeight the physical height of pattern, in user defined unit.
    @showExtration whether show extracted features and feature filtering.
    @nMiniMatches minimal number of matched features for a frame.
	@flags Calibration flags
    @criteria optimization stopping criteria.
    @detector feature detector that detect feature points in pattern and images.
    @descriptor feature descriptor.
    @matcher feature matcher.
    */
    MyMultiCameraCalibration(const std::vector<std::string> &cameraSerials,int cameraType, int nCameras, const std::string& fileName,
    const std::string& cameraConfigFolder, const std::string a_DoubleSideConfig ,cv::Size frontPatternSize, cv::Size backPatternSize,
     float patternWidth,
        float patternHeight, int verbose = 0, int showExtration = 0, int nMiniMatches = 20, int flags = 0,
        TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-7)
    );

    /* @brief load images
    */
    virtual void loadImages(const std::set<std::string> &outliers = std::set<std::string> ());

    /* @brief initialize multiple camera calibration. It calibrates each camera individually.
    */

    virtual void initialize();
    /* @brief optimization extrinsic parameters
    */
   
   std::set<std::string>  removeOutlier();
   virtual void writeParameters(const std::string& filename);
protected:


    std::string  cameraConfigFolder;
    std::vector<std::string> cameraSerials;
    std::string dataFolder;
    std::set<std::string> m_outliers;
    cv::Size _FrontPatternSize, _BackPatternSize;
    //the secondPattern with respect to the firstPattern, filename "doubleSideTransform.yaml"
    // PoseFront =  PoseBack * doubleSideTransform ;
    //PoseBack = PoseFront * doubleSideTransform;
    cv::Mat doubleSideTransform;
    cv::Mat doubleSideTransform_rvec, doubleSideTransform_tvec;

     cv::Mat doubleSideTransformInv;
    cv::Mat doubleSideTransform_rvecInv, doubleSideTransform_tvecInv;   

protected:
    bool isBackPattern(const cv::Mat & imgPoints);
    void readCornersTo(const std::string &filepath, int cameraVertex, cv::Mat &imagePoints,
        cv::Mat & objectPoints);

    void calcPatternPose(int cameraVertex, const cv::Mat &imagePoints,
        const cv::Mat & objectPoints,  cv::Mat &rvec, cv::Mat &tvec);

    int readTimestamps(const std::string &filepath, int cameraVertex);

    void loadOneSerial(int cameraVertex, const std::string&serial, const std::string&parfolder);

    virtual bool findTimStamp( int cameraVertex, int a_timestamp, int numPatternPoints);

    void readcameraIntrinsics();

    virtual void storeReaded(const std::string &filepath, int cameraVertex,  
        int timestamp, cv::Mat &imagePoints,
        cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec);

    void writeParameters2config();

    int transformBackPattern(cv::Mat &objectPoints);

    int transformBackPatternPose(const cv::Mat &objectPoints,cv::Mat &rvec, cv::Mat &tvec);

    virtual void identifyMultiCameraTimestamps();

    void storeReadedImp(const std::string &filepath, int cameraVertex,  
            int timestamp, cv::Mat &imagePoints,
            cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec);

    void readDoubleSide(const std::string a_DoubleSideConfig);

    void doublesideTransform2vec();

    virtual void computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE, Mat&deltaX);

    virtual void computePhotoCameraJacobian(int patternSide, const Mat& RvecPhoto, const Mat& TvecPhoto, const Mat& RvecCamera,
        const Mat& TvecCamera, const Mat& RvecDoubleside, const Mat& TvecDoubleside, 
        Mat& Rvectran, Mat& Tvectran,
        const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
        const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianDoubleside, Mat& E);

    virtual double computeProjectError(Mat& parameters);



};





//! @}

}} // namespace multicalib, cv
#endif