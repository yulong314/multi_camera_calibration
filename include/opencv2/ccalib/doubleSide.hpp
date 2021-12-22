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

#ifndef __OPENCV_DoubleSideCALIBRATION_HPP__
#define __OPENCV_DoubleSideCALIBRATION_HPP__


#include <string>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <filesystem>

#include "mymulticalib.hpp"

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

class CV_EXPORTS DoubleSideCalibration: public MyMultiCameraCalibration
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
    DoubleSideCalibration(const std::vector<std::string> &cameraSerials,
        int cameraType, int nCameras, const std::string& fileName,
        const std::string& cameraConfigFolder,  cv::Size frontPatternSize, cv::Size backPatternSize, 
        float patternWidth,
        float patternHeight, int verbose = 0, int showExtration = 0, int nMiniMatches = 20, int flags = 0,
        TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-8)
    );

    /* @brief load images
    */

    
    /* @brief initialize multiple camera calibration. It calibrates each camera individually.
    */


    /* @brief optimization extrinsic parameters
    */
   virtual void initialize();
   virtual void writeParameters(const std::string& filename);



protected:
    std::vector<cv::Mat> camerasPose;
    std::vector<cv::Mat> camerasPose_rvec;
    std::vector<cv::Mat> camerasPose_tvec;
protected:
    void loadCameraPose();

    void writeParameters2config();

    void cameraPose2vec();

    virtual void computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE, Mat&deltaX);

    virtual cv::Mat buildParas();

    void writeDoubleSideTransform();

    void initializeDoublesideTransform();

    int findAtimeStampThathave2camerasSee( );

    void twoEdgesOfTimestamp(int timestamp, int edgeidxes[]);

    cv::Mat findTransformOfTwoEdge(int edgeidxes[]);

    void sortEdgePair(int edgeidxes[]);

    virtual bool findTimStamp( int cameraVertex, int a_timestamp, int numPatternPoints);

    virtual void storeReaded(const std::string &filepath, int cameraVertex,  
        int timestamp, cv::Mat &imagePoints,
        cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec);
    virtual void computePhotoCameraJacobian(int patternSide, const Mat& rvecPhoto, const Mat& tvecPhoto, const Mat& rvecCamera,
    const Mat& tvecCamera, const Mat& rvecDoubleside, const Mat& tvecDoubleside, 
    Mat& rvecTran, Mat& tvecTran,
    const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
    const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianDoubleside, Mat& E);

    int getPhotoVertexParameters(int photoVertex);

    virtual void paras2vertex(const cv::Mat &extrinParam);

    virtual double computeProjectError(Mat& parameters);

    virtual void vector2parameters(const Mat& parameters, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs);



};





//! @}

}} // namespace multicalib, cv
#endif