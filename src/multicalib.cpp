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
#include "multicalib.hpp"
#include "opencv2/core.hpp"
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <chrono>
#include <cassert>
#include <Eigen/QR>
#include <Eigen/IterativeLinearSolvers>
#include <opencv2/core/eigen.hpp>
using namespace Eigen;
namespace cv { namespace multicalib {

MultiCameraCalibration::MultiCameraCalibration(int cameraType, int nCameras, const std::string& fileName,
    float patternWidth, float patternHeight, int verbose, int showExtration, int nMiniMatches, int flags, TermCriteria criteria,
    Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> descriptor,
    Ptr<DescriptorMatcher> matcher)
{
    _camType = cameraType;
    _nCamera = nCameras;
    _flags = flags;
    _nMiniMatches = nMiniMatches;
    _filename = fileName;
    _patternWidth = patternWidth;
    _patternHeight = patternHeight;
    _criteria = criteria;
    _showExtraction = showExtration;
    _objectPointsForEachCamera.resize(_nCamera);
    _imagePointsForEachCamera.resize(_nCamera);
    _cameraMatrix.resize(_nCamera);
    _distortCoeffs.resize(_nCamera);
    _xi.resize(_nCamera);
    _omEachCamera.resize(_nCamera);
    _tEachCamera.resize(_nCamera);
    _detector = detector;
    _descriptor = descriptor;
    _matcher = matcher;
	_verbose = verbose;
    for (int i = 0; i < _nCamera; ++i)
    {
        _vertexList.push_back(vertex());
    }
}


bool MultiCameraCalibration::isValidPose(const cv::Mat& tvec){
    auto x = tvec.at<float>(0);
    auto y = tvec.at<float>(1);
    auto z = tvec.at<float>(2);
    auto r = sqrt(x*x + y*y + z*z);
    return r < 3000 and r > 300;
}
bool MultiCameraCalibration::isValidPose(const cv::Vec3f & tvecVertex){
    auto x = tvecVertex[0];
    auto y = tvecVertex[1];
    auto z = tvecVertex[2];
    auto r = sqrt(x*x + y*y + z*z);
    bool isvalid = r < 3000 and r > 300;
    static int notvalidCnt = 0;
    if (!isvalid) {
        notvalidCnt ++;
        std::cout << "notvalidCnt:"<< notvalidCnt<< std::endl;
    }
    return isvalid;
}
double MultiCameraCalibration::run()
{
    loadImages();
    initialize();
    double error = optimizeExtrinsics();
    return error;
}
void MultiCameraCalibration::reset() {
    _edgeList.clear();
    _vertexList.clear();
    for (int i = 0; i < _nCamera; ++i)
    {
        _vertexList.push_back(vertex());
    }
    for (int i=0; i < _nCamera; ++i){
        filesEachCameraFull[i].clear();
        timestampFull[i].clear();
        timestampAvailable[i].clear();
        _objectPointsForEachCamera[i].clear();
        _imagePointsForEachCamera[i].clear();

        _omEachCamera[i].clear();
        _tEachCamera[i].clear();        
    }

}
cv::Mat MultiCameraCalibration::transform2colomn2(const cv::Mat & p){
    cv::Mat points;
    points.create(p.rows, 2,  CV_32F);
    for (int r = 0 ; r < p.rows; r++){
        
        cv::Point2f s = p.at<cv::Point2f>(r,0); 
        points.at<float>(r, 0) = s.x;
        points.at<float>(r, 1) = s.y;

    }

    return points;
}

std::vector<std::string> MultiCameraCalibration::readStringList()
{
    std::vector<std::string> l;
    l.resize(0);
    FileStorage fs(_filename, FileStorage::READ);

    FileNode n = fs.getFirstTopLevelNode();

    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((std::string)*it);

    return l;
}

void MultiCameraCalibration::loadImages()
{
    std::vector<std::string> file_list;
    file_list = readStringList();

    Ptr<FeatureDetector> detector = _detector;
    Ptr<DescriptorExtractor> descriptor = _descriptor;
    Ptr<DescriptorMatcher> matcher = _matcher;

    randpattern::RandomPatternCornerFinder finder(_patternWidth, _patternHeight, _nMiniMatches, CV_32F, _verbose, this->_showExtraction, detector, descriptor, matcher);
    Mat pattern = cv::imread(file_list[0]);
    finder.loadPattern(pattern);

    std::vector<std::vector<std::string> > filesEachCameraFull(_nCamera);
    std::vector<std::vector<int> > timestampFull(_nCamera);
	std::vector<std::vector<int> > timestampAvailable(_nCamera);

    for (int i = 1; i < (int)file_list.size(); ++i)
    {
        int cameraVertex, timestamp;
        std::string filename = file_list[i].substr(0, file_list[i].rfind('.'));
        size_t spritPosition1 = filename.rfind('/');
        size_t spritPosition2 = filename.rfind('\\');
        if (spritPosition1!=std::string::npos)
        {
            filename = filename.substr(spritPosition1+1, filename.size() - 1);
        }
        else if(spritPosition2!= std::string::npos)
        {
            filename = filename.substr(spritPosition2+1, filename.size() - 1);
        }
        sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
        filesEachCameraFull[cameraVertex].push_back(file_list[i]);
        timestampFull[cameraVertex].push_back(timestamp);
    }


    // calibrate each camera individually
    for (int camera = 0; camera < _nCamera; ++camera)
    {
        Mat image, cameraMatrix, distortCoeffs;

        // find image and object points
        for (int imgIdx = 0; imgIdx < (int)filesEachCameraFull[camera].size(); ++imgIdx)
        {
            image = imread(filesEachCameraFull[camera][imgIdx], IMREAD_GRAYSCALE);
			if (!image.empty() && _verbose)
			{
				std::cout << "open image " << filesEachCameraFull[camera][imgIdx] << " successfully" << std::endl;
			}
			else if (image.empty() && _verbose)
			{
				std::cout << "open image" << filesEachCameraFull[camera][imgIdx] << " failed" << std::endl;
			}
            std::vector<Mat> imgObj = finder.computeObjectImagePointsForSingle(image);
			if ((int)imgObj[0].total() > _nMiniMatches)
			{
				_imagePointsForEachCamera[camera].push_back(imgObj[0]);
				_objectPointsForEachCamera[camera].push_back(imgObj[1]);
				timestampAvailable[camera].push_back(timestampFull[camera][imgIdx]);
			}
			else if ((int)imgObj[0].total() <= _nMiniMatches && _verbose)
			{
				std::cout << "image " << filesEachCameraFull[camera][imgIdx] <<" has too few matched points "<< std::endl;
			}
        }

        // calibrate
        Mat idx;
        double rms = 0.0;
        if (_camType == PINHOLE)
        {
            rms = cv::calibrateCamera(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
                image.size(), _cameraMatrix[camera], _distortCoeffs[camera], _omEachCamera[camera],
                _tEachCamera[camera],_flags);
            idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
            for (int i = 0; i < (int)idx.total(); ++i)
            {
                idx.at<int>(i) = i;
            }
        }
        //else if (_camType == FISHEYE)
        //{
        //    rms = cv::fisheye::calibrate(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
        //        image.size(), _cameraMatrix[camera], _distortCoeffs[camera], _omEachCamera[camera],
        //        _tEachCamera[camera], _flags);
        //    idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
        //    for (int i = 0; i < (int)idx.total(); ++i)
        //    {
        //        idx.at<int>(i) = i;
        //    }
        //}
        else if (_camType == OMNIDIRECTIONAL)
        {
            rms = cv::omnidir::calibrate(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
                image.size(), _cameraMatrix[camera], _xi[camera], _distortCoeffs[camera], _omEachCamera[camera],
                _tEachCamera[camera], _flags, TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 1e-7),
                idx);
        }
        _cameraMatrix[camera].convertTo(_cameraMatrix[camera], CV_32F);
        _distortCoeffs[camera].convertTo(_distortCoeffs[camera], CV_32F);
        _xi[camera].convertTo(_xi[camera], CV_32F);
        //else
        //{
        //    CV_Error_(CV_StsOutOfRange, "Unknown camera type, use PINHOLE or OMNIDIRECTIONAL");
        //}

        for (int i = 0; i < (int)_omEachCamera[camera].size(); ++i)
        {
			int cameraVertex, timestamp, photoVertex;
			cameraVertex = camera;
			timestamp = timestampAvailable[camera][idx.at<int>(i)];

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

			this->_edgeList.push_back(edge(cameraVertex, photoVertex, idx.at<int>(i), transform));
        }
		std::cout << "initialized for camera " << camera << " rms = " << rms << std::endl;
		std::cout << "initialized camera matrix for camera " << camera << " is" << std::endl;
		std::cout << _cameraMatrix[camera] << std::endl;
		std::cout << "xi for camera " << camera << " is " << _xi[camera] << std::endl;
    }

}

int MultiCameraCalibration::getPhotoVertex(int timestamp)
{
    int photoVertex = INVALID;

    // find in existing photo vertex
    for (int i = 0; i < (int)_vertexList.size(); ++i)
    {
        if (_vertexList[i].timestamp == timestamp)
        {
            photoVertex = i;
            _vertexList[i].timestampCnt ++;
            break;
        }
    }

    // add a new photo vertex
    if (photoVertex == INVALID)
    {
        _vertexList.push_back(vertex(Mat::eye(4, 4, CV_32F), timestamp));
        photoVertex = (int)_vertexList.size() - 1;
    }

    return photoVertex;
}

void MultiCameraCalibration::simplifyPhotoVertexs(int beginVertex){
    std::vector<vertex> vertexList1;
    
    for (size_t i = 0; i < _vertexList.size(); i++)
    {
        vertex v= _vertexList[i];
        if (v.timestampCnt > 1 || i < beginVertex || v.timestamp < 0){
            vertexList1.push_back(v);
        }
    }

    _vertexList = vertexList1;
}
cv::Mat MultiCameraCalibration::buildGraph(){
    int nVertices = (int)_vertexList.size();
    int nEdges = (int) _edgeList.size();
    std::cout <<"edge size:"<<nEdges << std::endl;
    // build graph
    Mat G = Mat::zeros(nVertices, nVertices, CV_32S);
    for (int edgeIdx = 0; edgeIdx < nEdges; ++edgeIdx)
    {
        int cameraVertex = this->_edgeList[edgeIdx].cameraVertex;
        int photoVertex = this->_edgeList[edgeIdx].photoVertex;
        G.at<int>(cameraVertex, photoVertex) = edgeIdx + 1;
    }
    G = G + G.t();
    return G;
}

void MultiCameraCalibration::initCameraPose(){

}
void MultiCameraCalibration::initialize()
{
    // simplifyPhotoVertexs(_nCamera);
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
        Mat prePose = this->_vertexList[pre[vertexIdx]].pose;
        int edgeIdx = G.at<int>(vertexIdx, pre[vertexIdx]) - 1;
        Mat transform = this->_edgeList[edgeIdx].transform;

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
            //cameraPose * photoPose =  patternPoseInCamera;
            this->_vertexList[vertexIdx].pose = prePose.inv() * transform;
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
        }
    }
}

cv::Mat MultiCameraCalibration::buildParas(){
    // get om, t vector
    int nVertex = (int)this->_vertexList.size();

    Mat extrinParam(1, (nVertex-1)*6, CV_32F);
    int offset = 0;
    // the pose of the vertex[0] is eye
    for (int i = 1; i < nVertex; ++i)
    {
        Mat rvec, tvec;
        cv::Rodrigues(this->_vertexList[i].pose.rowRange(0,3).colRange(0, 3), rvec);
        this->_vertexList[i].pose.rowRange(0,3).col(3).copyTo(tvec);

        rvec.reshape(1, 1).copyTo(extrinParam.colRange(offset, offset + 3));
        tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
        offset += 6;
    }    
    return extrinParam;
}

void MultiCameraCalibration::paras2vertex(const cv::Mat &extrinParam){
    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(extrinParam, rvecVertex, tvecVertex);
    for (int verIdx = 1; verIdx < (int)_vertexList.size(); ++verIdx)
    {
        Mat R;
        Mat pose = Mat::eye(4, 4, CV_32F);
        Rodrigues(rvecVertex[verIdx-1], R);
        R.copyTo(pose.colRange(0, 3).rowRange(0, 3));
        Mat(tvecVertex[verIdx-1]).reshape(1, 3).copyTo(pose.rowRange(0, 3).col(3));
        _vertexList[verIdx].pose = pose;
		if (_verbose && verIdx < _nCamera)
		{
			std::cout << "final camera pose of camera " << verIdx << " is" << std::endl;
			std::cout << pose << std::endl;
		}
    }
}


double MultiCameraCalibration::optimizeExtrinsics()
{
    Mat extrinParam = buildParas();
    //double error_pre = computeProjectError(extrinParam);
    // optimization
    
    double change = 1;
    double normG = 0.0;
    double normExtrin = 0;
    Mat JTJ_inv, JTError;
    Mat G;
    for(int iter = 0; ; ++iter)
    {
        if ((_criteria.type == 1 && iter >= _criteria.maxCount)  ||
            (_criteria.type == 2 && change <= _criteria.epsilon) ||
            (_criteria.type == 3 && (change <= _criteria.epsilon || iter >= _criteria.maxCount))){
                int ss = 0;
                break;
            }
#if 1            
        const double alpha_smooth = 0.95;
        double alpha_smooth2 = std::pow(alpha_smooth, (double)iter + 1.0) ;
#else
        const double alpha_smooth = 0.01;
        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, (double)iter + 1.0);
#endif
        cv::Mat deltx;
        this->computeJacobianExtrinsic(extrinParam, JTJ_inv, JTError, deltx);
        // G = alpha_smooth2*JTJ_inv * JTError;
        G = alpha_smooth2 * deltx;
        std::cout <<"alpha_smooth2:" << alpha_smooth2<<" "<< std::endl;
        if (G.depth() == CV_64F)
        {
            G.convertTo(G, CV_32F);

        }
        auto Gt = G.reshape(1, 1);
        std::cout <<"extrinParam:" <<extrinParam << std::endl;
        std::cout <<"Gt:" << Gt << std::endl;
        extrinParam = extrinParam + Gt;
        normG = norm(G);
        normExtrin = norm(extrinParam);
        change = norm(G) / norm(extrinParam);
        //double error = computeProjectError(extrinParam);
        std::cout << "iter:"<<iter <<"change:"<<change<<std::endl;
    }

    double error = computeProjectError(extrinParam);


    paras2vertex(extrinParam);
    return error;
}

template<typename MatrixType> void qr(const MatrixType& m, Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>& q, 
Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> & r)
{
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
//   typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;

//   MatrixType a = MatrixType::Random(rows,cols);
  HouseholderQR<MatrixType> qrOfA(m);

  q = qrOfA.householderQ();
//   VERIFY_IS_UNITARY(q);

  r = qrOfA.matrixQR().template triangularView<Upper>();
//   VERIFY_IS_APPROX(m, q * r);

}
void qRDecomposition(Eigen::MatrixXd &A,Eigen::MatrixXd &Q, Eigen::MatrixXd &R)
{
    /*
        A=QR
        Q: is orthogonal matrix-> columns of Q are orthonormal
        R: is upper triangulate matrix

        this is possible when columns of A are linearly indipendent

    */

    Eigen::MatrixXd thinQ(A.rows(),A.cols() ), q(A.rows(),A.rows());

    Eigen::HouseholderQR<Eigen::MatrixXd> householderQR(A);
    q = householderQR.householderQ();
    thinQ.setIdentity();
    Q = householderQR.householderQ() * thinQ;
    R=Q.transpose()*A;
}

void QRdecompose(const cv::Mat &J, cv::Mat &Q, cv::Mat &R){
    MatrixXd eJ(J.rows, J.cols);
    cv::cv2eigen(J, eJ);
    // std::cout <<"eJ" <<eJ <<std::endl;
    MatrixXd eQ, eR;
    qRDecomposition(eJ, eQ, eR);

    cv::eigen2cv(eQ, Q);
    cv::eigen2cv(eR, R);
}
void MultiCameraCalibration::sparseSolver(const MatrixXd&a_A, Eigen::MatrixXd&b, Eigen::MatrixXd&x){
    // VectorXd x(n), b(n);
    // SparseMatrix<double> A(n,n);
    // fill A and b
    SparseMatrix<double> A = a_A.sparseView();
    // std::cout << "sparse A"<< A<<std::endl;
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
    cg.compute(A);
    x = cg.solve(b);
    // std::cout << "#iterations:     " << cg.iterations() << std::endl;
    // std::cout << "estimated error: " << cg.error()      << std::endl;
    // update b, and solve again
    x = cg.solve(b);

}
cv::Mat MultiCameraCalibration::conjungate(const cv::Mat &a, const cv::Mat &b){
    cv::Mat x;
    MatrixXd eA(a.rows, a.cols);
    MatrixXd eb(b.rows, b.cols);
    MatrixXd ex;
    cv::cv2eigen(a, eA);
    cv::cv2eigen(b, eb);
    sparseSolver(eA, eb,ex);
    cv::eigen2cv(ex, x);
    return x;


}
void MultiCameraCalibration::computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE, Mat&deltaX)
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

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int photoVertex = _edgeList[edgeIdx].photoVertex;
        int photoIndex = _edgeList[edgeIdx].photoIndex;
        int cameraVertex = _edgeList[edgeIdx].cameraVertex;

        Mat objectPoints = _objectPointsForEachCamera[cameraVertex][photoIndex];
        Mat imagePoints = _imagePointsForEachCamera[cameraVertex][photoIndex];

        Mat rvecTran, tvecTran;
        Mat R = _edgeList[edgeIdx].transform.rowRange(0, 3).colRange(0, 3);
        tvecTran = _edgeList[edgeIdx].transform.rowRange(0, 3).col(3);
        cv::Rodrigues(R, rvecTran);
        assert(isValidPose(tvecTran));

        //为什么是Photovertex,而不是photoIndex
        Mat rvecPhoto = extrinsicParams.colRange((photoVertex-1)*6, (photoVertex-1)*6 + 3);
        Mat tvecPhoto = extrinsicParams.colRange((photoVertex-1)*6 + 3, (photoVertex-1)*6 + 6);
        assert(isValidPose(tvecPhoto));
        Mat rvecCamera, tvecCamera;
        if (cameraVertex > 0)
        {
            rvecCamera = extrinsicParams.colRange((cameraVertex-1)*6, (cameraVertex-1)*6 + 3);
            tvecCamera = extrinsicParams.colRange((cameraVertex-1)*6 + 3, (cameraVertex-1)*6 + 6);
        }
        else
        {
            rvecCamera = Mat::zeros(3, 1, CV_32F);// 第一个相机姿态无旋转平移
            tvecCamera = Mat::zeros(3, 1, CV_32F);
        }

        Mat jacobianPhoto, jacobianCamera, error, error1;

        // const float epsilon = 0.001;
        // cv::Mat tvecCameraEpsilon = tvecCamera.clone();
        // tvecCameraEpsilon += epsilon;


        // computePhotoCameraJacobian(rvecPhoto, tvecPhoto, rvecCamera, tvecCameraEpsilon, rvecTran, tvecTran,
        //     objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
        //     this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error1);

        computePhotoCameraJacobian(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
            objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
            this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error);

            // cv::Mat errordiff = error1 - error;
            // cv::Mat difference =errordiff / epsilon;
            // cv::Mat derivative = jacobianCamera.col(3);
            // cv::Mat accuracy = difference - derivative;
            // std::cout << "tvecCameraEpsilon" << tvecCameraEpsilon << std::endl;
            // std::cout << "tvecCamera" << tvecCamera << std::endl;

            // std::cout << "jacobianCamera" << jacobianCamera << std::endl;
            // std::cout << "difference" << difference.t() << std::endl;
            // std::cout << "derivative" << derivative.t() << std::endl;
            // std::cout << "accuracy" << accuracy.t() << std::endl;
            // int kk1 = 1;

        if (cameraVertex > 0)
        {
            jacobianCamera.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
                colRange((cameraVertex-1)*6, cameraVertex*6));
        }
        jacobianPhoto.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
            colRange((photoVertex-1)*6, photoVertex*6));
        error.copyTo(E.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]));
    }
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
    deltaX = x;
    std::cout <<"x:" << x.t()<< std::endl;
    auto end1 = std::chrono::steady_clock::now();
    auto elpased1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count();   
    // std::cout << "inv taks milliseconds:"<< elpased1 << std::endl; 
    // JTJ_inv = JTJ.inv();
    // auto end = std::chrono::steady_clock::now();
    // auto elpased = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "inv taks milliseconds:"<< elpased << std::endl;
    std::cout << "normE:"<< norm(E) << std::endl;

}
bool MultiCameraCalibration::IsvalidImagePoints(const Mat& imagePoints2){

    for (int r =0; r < imagePoints2.rows; r++){
        assert(imagePoints2.cols == 2);
        auto x = imagePoints2.at<float>(r, 0);
        auto y = imagePoints2.at<float>(r, 1);
        assert(x >=0 && y>=0);
        assert(x < 1920 && y < 1080);
    }
    return true;

}

void MultiCameraCalibration::computePhotoCameraJacobian(const Mat& rvecPhoto, const Mat& tvecPhoto, const Mat& rvecCamera,
    const Mat& tvecCamera, Mat& rvecTran, Mat& tvecTran, const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
    const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianCamera, Mat& E)
{
    Mat drvecTran_drecvPhoto, drvecTran_dtvecPhoto,
        drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto,
        dtvecTran_drvecCamera, dtvecTran_dtvecCamera;

    // const float epsilon = 0.1;
    // cv::Mat rvecCamera1 = tvecCamera.clone();
    // rvecCamera1.at<float>(2) += epsilon;
    // Mat  tvecTran1;
    // compose_motion(rvecPhoto, tvecPhoto, rvecCamera1, tvecCamera, rvecTran, tvecTran1,
    //     drvecTran_drecvPhoto, drvecTran_dtvecPhoto, drvecTran_drvecCamera, drvecTran_dtvecCamera,
    //     dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto, dtvecTran_drvecCamera, dtvecTran_dtvecCamera);

    compose_motion(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
        drvecTran_drecvPhoto, drvecTran_dtvecPhoto, drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto, dtvecTran_drvecCamera, dtvecTran_dtvecCamera);
    
    // auto diff1 = (tvecTran1 - tvecTran)/epsilon;
    // std::cout << "diff:"<< diff1 << std::endl;
    // std::cout << "dtvecTran_drvecCamera:"<< dtvecTran_drvecCamera << std::endl;
 
    if (rvecTran.depth() == CV_64F)
    {
        rvecTran.convertTo(rvecTran, CV_32F);
    }
    if (tvecTran.depth() == CV_64F)
    {
        tvecTran.convertTo(tvecTran, CV_32F);
    }
    float xif = 0.0f;
    if (_camType == OMNIDIRECTIONAL)
    {
        xif= xi.at<float>(0);
    }

    Mat imagePoints2, jacobian, dx_drvecCamera, dx_dtvecCamera, dx_drvecPhoto, dx_dtvecPhoto;
    if (_camType == PINHOLE)
    {
        // cv::Mat tvecTran1 = tvecTran.clone();
        // tvecTran1.at<float>(0) += epsilon;
        // // cv::Mat K1 = K.clone();
        // std::cout <<"tvecTran:"<< tvecTran << std::endl;
 
        // std::cout <<"tvecTran1:"<< tvecTran1 << std::endl;

        // cv::Mat imagePoints2Esp;
        // cv::projectPoints(objectPoints, rvecTran, tvecTran1, K, distort, imagePoints2Esp, jacobian);
        // imagePoints2Esp = transform2colomn2(imagePoints2Esp);


        cv::projectPoints(objectPoints, rvecTran, tvecTran, K, distort, imagePoints2, jacobian);
        imagePoints2 = transform2colomn2(imagePoints2);
    //     auto diff1 = (imagePoints2Esp - imagePoints2)/epsilon;
    //     std::cout << " imagePoints2Esp:"<< imagePoints2Esp << std::endl;
    //     std::cout << " imagePoints2:"<< imagePoints2 << std::endl;
    //     std::cout << "diff :"<< diff1 << std::endl;
    //     std::cout << "jacobian.col(3):"<< jacobian.col(3) << std::endl;
    //     std::cout << "jacobian:"<< jacobian << std::endl;
    //     assert (IsvalidImagePoints(imagePoints2));
    // }
    //else if (_camType == FISHEYE)
    //{
    //    cv::fisheye::projectPoints(objectPoints, imagePoints2, rvecTran, tvecTran, K, distort, 0, jacobian);
    }
    else if (_camType == OMNIDIRECTIONAL)
    {
        cv::omnidir::projectPoints(objectPoints, imagePoints2, rvecTran, tvecTran, K, xif, distort, jacobian);
    }
    if (objectPoints.depth() == CV_32F)
    {
        Mat(imagePoints - imagePoints2).convertTo(E, CV_64FC2);
    }
    else
    {
        E = imagePoints - imagePoints2;
    }
    E = E.reshape(1, (int)imagePoints.rows*2);

    dx_drvecCamera = jacobian.colRange(0, 3) * drvecTran_drvecCamera + jacobian.colRange(3, 6) * dtvecTran_drvecCamera;
    dx_dtvecCamera = jacobian.colRange(0, 3) * drvecTran_dtvecCamera + jacobian.colRange(3, 6) * dtvecTran_dtvecCamera;

    if (1){
        std::cout<< "jacobian.colRange(0, 3):"<< jacobian.colRange(0, 3) << std::endl;
        std::cout<< "jacobian.colRange(3, 6):"<< jacobian.colRange(3, 6) << std::endl;
        std::cout<<"drvecTran_drvecCamera:" << drvecTran_drvecCamera<< std::endl;
        std::cout<<"dtvecTran_drvecCamera:" << dtvecTran_drvecCamera<< std::endl;
        std::cout<<"drvecTran_dtvecCamera:" << drvecTran_dtvecCamera<< std::endl;
        std::cout<<"dtvecTran_dtvecCamera:" << dtvecTran_dtvecCamera<< std::endl;
        std::cout<<"dx_dtvecCamera:" << dx_dtvecCamera<< std::endl;
        std::cout<<"dx_drvecCamera:" << dx_drvecCamera<< std::endl;
        int kkk=1;
    }    
    dx_drvecPhoto = jacobian.colRange(0, 3) * drvecTran_drecvPhoto + jacobian.colRange(3, 6) * dtvecTran_drvecPhoto;
    dx_dtvecPhoto = jacobian.colRange(0, 3) * drvecTran_dtvecPhoto + jacobian.colRange(3, 6) * dtvecTran_dtvecPhoto;

    jacobianCamera = cv::Mat(dx_drvecCamera.rows, 6, CV_64F);
    jacobianPhoto = cv::Mat(dx_drvecPhoto.rows, 6, CV_64F);

    dx_drvecCamera.copyTo(jacobianCamera.colRange(0, 3));
    dx_dtvecCamera.copyTo(jacobianCamera.colRange(3, 6));
    dx_drvecPhoto.copyTo(jacobianPhoto.colRange(0, 3));
    dx_dtvecPhoto.copyTo(jacobianPhoto.colRange(3, 6));

}
void MultiCameraCalibration::graphTraverse(const Mat& G, int begin, std::vector<int>& order, std::vector<int>& pre)
{
    CV_Assert(!G.empty() && G.rows == G.cols);
    int nVertex = G.rows;
    order.resize(0);
    pre.resize(nVertex, INVALID);
    pre[begin] = -1;
    std::vector<bool> visited(nVertex, false);
    std::queue<int> q;
    visited[begin] = true;
    q.push(begin);
    order.push_back(begin);

    while(!q.empty())
    {
        int v = q.front();
        q.pop();
        Mat idx;
        // use my findNonZero maybe
        findRowNonZero(G.row(v), idx);
        for(int i = 0; i < (int)idx.total(); ++i)
        {
            int neighbor = idx.at<int>(i);
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                q.push(neighbor);
                order.push_back(neighbor);
                pre[neighbor] = v;
            }
        }
    }
}

void MultiCameraCalibration::findRowNonZero(const Mat& row, Mat& idx)
{
    CV_Assert(!row.empty() && row.rows == 1 && row.channels() == 1);
    Mat _row;
    std::vector<int> _idx;
    row.convertTo(_row, CV_32F);
    for (int i = 0; i < (int)row.total(); ++i)
    {
        if (_row.at<float>(i) != 0)
        {
            _idx.push_back(i);
        }
    }
    idx.release();
    idx.create(1, (int)_idx.size(), CV_32S);
    for (int i = 0; i < (int)_idx.size(); ++i)
    {
        idx.at<int>(i) = _idx[i];
    }
}

void MultiCameraCalibration::sortedgelist(const std::vector<edge>& ori_edgelist,std::vector<edge> &a_edgelist){
    struct {
        bool operator()(const edge& a, const edge & b) const { return a.reprojecterror > b.reprojecterror; }
    } customLess;    
    a_edgelist = ori_edgelist;
    std::sort(a_edgelist.begin(), a_edgelist.end(),customLess);

}

void MultiCameraCalibration::printedgelist(const std::vector<edge>& a_edgelist){
    for (int i = 0; i <  a_edgelist.size();  i++){
        const edge &edge1 = a_edgelist[i];
        std::cout << edge1.reprojecterror << ":" << filesEachCameraFull[edge1.cameraVertex][edge1.photoIndex] << std::endl;
    }
}
double MultiCameraCalibration::computeProjectError(Mat& parameters)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.total() == (nVertex-1) * 6 && parameters.depth() == CV_32F);
    int nEdge = (int)_edgeList.size();

    // recompute the transform between photos and cameras

    std::vector<edge> edgeList = this->_edgeList;
    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(parameters, rvecVertex, tvecVertex);

    float totalError = 0;
    int totalNPoints = 0;
    std::vector<float> errorsVector;
    std::vector<float> errorsVectorPerImage;
    
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat RPhoto, RCamera, TPhoto, TCamera, transform;
        int cameraVertex = edgeList[edgeIdx].cameraVertex;
        int photoVertex = edgeList[edgeIdx].photoVertex;
        int PhotoIndex = edgeList[edgeIdx].photoIndex;
        TPhoto = Mat(tvecVertex[photoVertex - 1]).reshape(1, 3);

        //edgeList[edgeIdx].transform = Mat::ones(4, 4, CV_32F);
        transform = Mat::eye(4, 4, CV_32F);
        cv::Rodrigues(rvecVertex[photoVertex-1], RPhoto);
        if (cameraVertex == 0)
        {
            RPhoto.copyTo(transform.rowRange(0, 3).colRange(0, 3));
            TPhoto.copyTo(transform.rowRange(0, 3).col(3));
        }
        else
        {
            TCamera = Mat(tvecVertex[cameraVertex - 1]).reshape(1, 3);
            cv::Rodrigues(rvecVertex[cameraVertex - 1], RCamera);
            Mat(RCamera*RPhoto).copyTo(transform.rowRange(0, 3).colRange(0, 3));
            Mat(RCamera * TPhoto + TCamera).copyTo(transform.rowRange(0, 3).col(3));
        }

        transform.copyTo(edgeList[edgeIdx].transform);
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
        edgeList[edgeIdx].reprojecterror = errorPerImage/error.rows;
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

void MultiCameraCalibration::compose_motion(InputArray _om1, InputArray _T1, 
    InputArray _om2, InputArray _T2, Mat& om3, Mat& T3, Mat& dom3dom1,
    Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2)
{
    Mat om1, om2, T1, T2;
    _om1.getMat().convertTo(om1, CV_64F);
    _om2.getMat().convertTo(om2, CV_64F);
    _T1.getMat().reshape(1, 3).convertTo(T1, CV_64F);
    _T2.getMat().reshape(1, 3).convertTo(T2, CV_64F);
    /*Mat om2 = _om2.getMat();
    Mat T1 = _T1.getMat().reshape(1, 3);
    Mat T2 = _T2.getMat().reshape(1, 3);*/

    //% Rotations:
    Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
    cv::Rodrigues(om1, R1, dR1dom1);
    cv::Rodrigues(om2, R2, dR2dom2);
    /*JRodriguesMatlab(dR1dom1, dR1dom1);
    JRodriguesMatlab(dR2dom2, dR2dom2);*/
    dR1dom1 = dR1dom1.t();
    dR2dom2 = dR2dom2.t();

    R3 = R2 * R1;
    Mat dR3dR2, dR3dR1;
    //dAB(R2, R1, dR3dR2, dR3dR1);
    matMulDeriv(R2, R1, dR3dR2, dR3dR1);
    Mat dom3dR3;
    cv::Rodrigues(R3, om3, dom3dR3);
    //JRodriguesMatlab(dom3dR3, dom3dR3);
    dom3dR3 = dom3dR3.t();

    dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
    dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
    dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
    dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

    //% Translations:
    Mat T3t = R2 * T1;
    Mat dT3tdR2, dT3tdT1;
    //dAB(R2, T1, dT3tdR2, dT3tdT1);
    matMulDeriv(R2, T1, dT3tdR2, dT3tdT1);

    Mat dT3tdom2 = dT3tdR2 * dR2dom2;
    T3 = T3t + T2;
    dT3dT1 = dT3tdT1;
    dT3dT2 = Mat::eye(3, 3, CV_64FC1);
    dT3dom2 = dT3tdom2;
    dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
}

void MultiCameraCalibration::vector2parameters(const Mat& parameters, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs)
{

    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.channels() == 1 && (int)parameters.total() == 6*(nVertex - 1));
    CV_Assert(parameters.depth() == CV_32F);
    parameters.reshape(1, 1);

    rvecVertex.reserve(0);
    tvecVertexs.resize(0);

    for (int i = 0; i < nVertex - 1; ++i)
    {
        rvecVertex.push_back(Vec3f(parameters.colRange(i*6, i*6 + 3)));
        tvecVertexs.push_back(Vec3f(parameters.colRange(i*6 + 3, i*6 + 6)));
        isValidPose(Vec3f(parameters.colRange(i*6 + 3, i*6 + 6)));
    }
}

void MultiCameraCalibration::parameters2vector(const std::vector<Vec3f>& rvecVertex, const std::vector<Vec3f>& tvecVertex, Mat& parameters)
{
    CV_Assert(rvecVertex.size() == tvecVertex.size());
    int nVertex = (int)rvecVertex.size();
    // the pose of the first camera is known
    parameters.create(1, 6*(nVertex-1), CV_32F);
    
    for (int i = 0; i < nVertex-1; ++i)
    {
        Mat(rvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6, i*6 + 3));
        Mat(tvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6 + 3, i*6 + 6));
        isValidPose(tvecVertex[i]);
    }
}

void MultiCameraCalibration::writeParameters(const std::string& filename)
{
    FileStorage fs( filename, FileStorage::WRITE );

    fs << "nCameras" << _nCamera;

    for (int camIdx = 0; camIdx < _nCamera; ++camIdx)
    {
        std::stringstream tmpStr;
        tmpStr << camIdx;
        std::string cameraMatrix = "camera_matrix_" + tmpStr.str();
        std::string cameraPose = "camera_pose_" + tmpStr.str();
        std::string cameraDistortion = "camera_distortion_" + tmpStr.str();
        std::string cameraXi = "xi_" + tmpStr.str();

        fs << cameraMatrix << _cameraMatrix[camIdx];
        fs << cameraDistortion << _distortCoeffs[camIdx];
        if (_camType == OMNIDIRECTIONAL)
        {
            fs << cameraXi << _xi[camIdx].at<float>(0);
        }

        fs << cameraPose << _vertexList[camIdx].pose;
    }

    fs << "meanReprojectError" <<_error;

    for (int photoIdx = _nCamera; photoIdx < (int)_vertexList.size(); ++photoIdx)
    {
        std::stringstream tmpStr;
        tmpStr << _vertexList[photoIdx].timestamp;
        std::string photoTimestamp = "pose_timestamp_" + tmpStr.str();

        fs << photoTimestamp << _vertexList[photoIdx].pose;
    }
}
}} // namespace multicalib, cv
