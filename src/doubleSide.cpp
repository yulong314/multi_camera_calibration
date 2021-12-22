
#include "doubleSide.hpp"
#include <opencv2/calib3d.hpp>
namespace cv{    namespace multicalib    {

DoubleSideCalibration::DoubleSideCalibration(const std::vector<std::string> &a_cameraSerials, 
    int cameraType, int nCameras, const std::string &a_dataFolder,
                const std::string &a_cameraConfigFolder, cv::Size frontPatternSize, cv::Size backPatternSize,
                float patternWidth,
                float patternHeight, int verbose, int showExtration, int nMiniMatches , int flags,
                TermCriteria criteria) : MyMultiCameraCalibration(a_cameraSerials, cameraType, nCameras,
                            a_dataFolder, a_cameraConfigFolder, "", frontPatternSize, backPatternSize,
                            patternWidth, patternHeight, verbose, showExtration, nMiniMatches, flags, criteria)
{
    loadCameraPose();
    _vertexList.clear();
    for (int i = 0; i < _nCamera; ++i)
    {
        camerasPose_rvec.resize(_nCamera);
        camerasPose_tvec.resize(_nCamera);
        vertex vertex1 = vertex(camerasPose[i], -1);
        
        _vertexList.push_back(vertex1);
    }  
    cameraPose2vec();  
}
int DoubleSideCalibration::findAtimeStampThathave2camerasSee( ){
    for( int i = 0; i < _vertexList.size(); i++){
        auto &v = _vertexList[i];
        if (v.timestampCnt > 1){
            return v.timestamp;
        }
    }
    assert(false);
    return -1;
}
void DoubleSideCalibration::twoEdgesOfTimestamp(int timestamp, int edgeidxes[]){
    int cnt = 0;
    for (int i = 0; i < _edgeList.size(); i++){
        auto eg = _edgeList[i];
        int photoVertex = eg.photoVertex;
        int egtimestamp =  _vertexList[photoVertex].timestamp;  
        if (egtimestamp == timestamp){
            assert(cnt < 2);
            edgeidxes[cnt] = i;
            cnt ++;
        }
        
    }
    assert (cnt == 2);
}

/*

let     : PinbackPatternCoord = doubleSideTransform * PinFrontPatternCoord;     (1)
because : PbackPatternInCamBack = backPose * PinbackPatternCoord; a point in backPattern transformed to point in camera1 coordinate
so      : PinbackPatternCoord = backPose.inv() * PbackPatternInCamBack;            (2)
similerly:
because : PfrontPatternInCamFront = frontPose * pinFrontPatternCoord;
so      : pinFrontPatternCoord = frontpose.inv()* PfrontPatternInCamFront;          (3)

let     :Pworld ,be a point in world coordinate system
because : PbackPatternInCamBack = camPoseBack * Pworld;                            (4)
          PfrontPatternInCamFront= camPoseFront * Pworld;                           (5)

        substitute 2~3 with 4~5:
        PinbackPatternCoord = backPose.inv() * camPoseBack * Pworld;            (6)
        pinFrontPatternCoord = frontpose.inv()* camPoseFront * Pworld;          (7)

        substitute (1) to (6):
        doubleSideTransform * PinFrontPatternCoord = backPose.inv() * camPoseBack * Pworld;     (8)

        from (7):
        Pworld = camPoseFront.inv() * frontPose * pinFrontPatternCoord ;                         (9)
        from(8):
        Pworld = camPoseBack.inv() * backPose * doubleSideTransform  * PinFrontPatternCoord;     (10)

        from (9),(10):
        camPoseBack.inv() * backPose * doubleSideTransform  * PinFrontPatternCoord =  camPoseFront.inv() * frontPose * pinFrontPatternCoord;  (11)

        from (11)
        camPoseBack.inv() * backPose * doubleSideTransform =  camPoseFront.inv() * frontPose;    (12)

        from (12)
        doubleSideTransform = backPose.inv() * camPoseBack *  camPoseFront.inv() *  frontPose ;    (13)

*/
void DoubleSideCalibration::sortEdgePair(int edgeidxes[]){
    int idx = edgeidxes[0];
    auto eg = _edgeList[idx];
    int photoIdx = eg.photoIndex;
    int cameraVertex = eg.cameraVertex;
    cv::Mat &objectPoints =_objectPointsForEachCamera[cameraVertex][photoIdx];
    if (isBackPattern(objectPoints)){
        std::swap(edgeidxes[0] , edgeidxes[1]);
    }
}


bool DoubleSideCalibration::findTimStamp( int cameraVertex, int a_timestamp, int numPatternPoints) {
        const std::vector<int> &ta = timestampAvailable[cameraVertex];
    for (int i = 0;i < ta.size(); i++){
        auto timestamp = ta[i];
        auto &imagepoints = _imagePointsForEachCamera[cameraVertex][i];
        int camPatternPoints  = imagepoints.rows;
        //find different pattern at the same timestamp
        if (timestamp == a_timestamp and camPatternPoints != numPatternPoints){
            return true;
        }
    }
    return false;
}

void DoubleSideCalibration::storeReaded(const std::string &filepath, int cameraVertex,  
        int timestamp, cv::Mat &imagePoints,
        cv::Mat & objectPoints, const cv::Mat &rvec, const cv::Mat &tvec) {
    MyMultiCameraCalibration::storeReadedImp(filepath, cameraVertex, timestamp, imagePoints, objectPoints, rvec, tvec);
}
cv::Mat DoubleSideCalibration::findTransformOfTwoEdge(int edgeidxes[]){
    cv::Mat patternPose[2];
    cv::Mat camPose[2];
    for (int i =0; i < 2; i++){
        auto &edg = _edgeList[edgeidxes[i]];
        patternPose[i] = edg.transform;
        auto cameraVertex = edg.cameraVertex;
        camPose[i] = camerasPose[cameraVertex];
        std::cout << "transform"<< i <<": " <<patternPose[i] << std::endl;
        std::cout << "camPose"<< i <<": " <<camPose[i] << std::endl;
    }
    cv::Mat backpose = camPose[1].inv() *patternPose[1] ;
    cv::Mat frontpose = camPose[0].inv() *patternPose[0] ;
    std::cout << "backpose"<< backpose << std::endl;
    std::cout << "frontpose"<< frontpose << std::endl;
    //frontposeInv = patternPose[0].inv() * camPose[0];
    //transform = patternPose[0].inv() * camPose[0] * camPose[1].inv() *patternPose[1];
    //transforminv = (camPose[1].inv() *patternPose[1]).inv() * (patternPose[0].inv() * camPose[0]).inv()
    // = patternPose[1].inv() * camPose[1] * camPose[0].inv() *  patternPose[0];
    //frontpose * transform = backpose
    //frontpose = backpose * transform.inv();
    cv::Mat transform1 = frontpose.inv() * backpose ;
    cv::Mat transform = patternPose[1].inv() *camPose[1] * camPose[0].inv() * patternPose[0];
    // std::cout << "Transform1: " << transform1 << std::endl;
    // std::cout << "Transform1.inv: " << transform1.inv() << std::endl;
    // std::cout << "Transform: " << transform << std::endl;

    return transform1;
}


void DoubleSideCalibration::initializeDoublesideTransform(){
    int timestamp ;
    timestamp = findAtimeStampThathave2camerasSee();
    int edgeidxes[2];
    twoEdgesOfTimestamp(timestamp, edgeidxes);
    int photoidx1 = _edgeList[edgeidxes[0]].photoIndex;
    int photoidx2 = _edgeList[edgeidxes[1]].photoIndex;
    int cameraVertex1 = _edgeList[edgeidxes[0]].cameraVertex;
    int cameraVertex2 = _edgeList[edgeidxes[1]].cameraVertex;   
    int rows1 =  _objectPointsForEachCamera[cameraVertex1][photoidx1].rows;
    int rows2 =  _objectPointsForEachCamera[cameraVertex2][photoidx2].rows;
    // assert (rows1 != rows2 );
    cv::Mat transform = findTransformOfTwoEdge(edgeidxes);
    doubleSideTransform = transform;
    doublesideTransform2vec();
}

void DoubleSideCalibration::initialize() {
    initializeDoublesideTransform();
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
            std::cout << "cameraBackPose" << camerasPose[1] << std::endl;
        }
        if (vertexIdx < _nCamera)
        {
            
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
// cv::Mat MultiCameraCalibration::buildParas(){
//     // get om, t vector
//     int nVertex = (int)this->_vertexList.size();

//     Mat extrinParam(1, (nVertex-1)*6, CV_32F);
//     int offset = 0;
//     // the pose of the vertex[0] is eye
//     for (int i = 1; i < nVertex; ++i)
//     {
//         Mat rvec, tvec;
//         cv::Rodrigues(this->_vertexList[i].pose.rowRange(0,3).colRange(0, 3), rvec);
//         this->_vertexList[i].pose.rowRange(0,3).col(3).copyTo(tvec);

//         rvec.reshape(1, 1).copyTo(extrinParam.colRange(offset, offset + 3));
//         tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
//         offset += 6;
//     }    
//     return extrinParam;
// }
cv::Mat DoubleSideCalibration::buildParas(){
    // get om, t vector
    int nVertex = (int)this->_vertexList.size();


    Mat extrinParam(1, (nVertex- _nCamera + 1)*6, CV_32F);
    int offset = 0;
    // the pose of the vertex[0] is eye

    doubleSideTransform_rvec.reshape(1, 1).copyTo(extrinParam.colRange(offset, offset + 3));
    doubleSideTransform_tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
    offset += 6;
    for (int i = _nCamera; i < nVertex; ++i)
    {
        int row = getPhotoVertexParameters(i);
        assert(row * 6 == offset);

        Mat rvec, tvec;
        auto &vertexPose = this->_vertexList[i].pose;
        cv::Rodrigues(vertexPose.rowRange(0,3).colRange(0, 3), rvec);
        vertexPose.rowRange(0,3).col(3).copyTo(tvec);
        Mat rreshape = rvec.reshape(1, 1);
        rreshape.copyTo(extrinParam.colRange(offset, offset + 3));
        tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
        offset += 6;

    }    
    return extrinParam;
}
void DoubleSideCalibration::cameraPose2vec(){
    for (int i = 0; i < _nCamera; i++){
        Mat R;
        camerasPose[i].colRange(0, 3).rowRange(0, 3).copyTo(R);
        cv::Mat i1 = R * R.t();

        Rodrigues(R, camerasPose_rvec[i]);
        
        camerasPose[i].rowRange(0, 3).col(3).copyTo(camerasPose_tvec[i]);
   
    }


}
void DoubleSideCalibration::loadCameraPose()
{
    for (auto serial : cameraSerials)
    {
        std::string filename = cameraConfigFolder + "/" + serial + ".xml";
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        cv::Mat pose;
        fs["CameraMatrix"] >> pose;
        // pose.convertTo(pose, CV_64F);
        camerasPose.push_back(pose);
    }
}
void DoubleSideCalibration::computePhotoCameraJacobian(int patternSide, const Mat& RvecPhoto, const Mat& TvecPhoto, const Mat& RvecCamera,
    const Mat& TvecCamera, const Mat& RvecDoubleside, const Mat& TvecDoubleside, 
    Mat& Rvectran, Mat& Tvectran,
    const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
    const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianDoubleside, Mat& E)
{
    //cameraPose * photoPose * doubleSideTransform = backpose;
    //cameraPose * photoPose  = frontPose; in camera coordinates
    Mat dRvectran_dRvecPhotofront, dRvectran_dTvecPhotofront,
        dTvectran_dRvecPhotofront, dTvectran_dTvecPhotofront,

        dRvectran_dRvecDoubleside, dRvectran_dTvecDoubleside,
        dTvectran_dRvecDoubleside, dTvectran_dTvecDoubleside;
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
    }else{
        dRvectran_dRvecPhoto =  dRvecPhotofront_dRvecPhoto;
        dRvectran_dTvecPhoto =  dRvecPhotofront_dTvecPhoto;
        dTvectran_dRvecPhoto =  dTvecPhotofront_dRvecPhoto;
        dTvectran_dTvecPhoto =  dTvecPhotofront_dTvecPhoto;    
        int rows = RvecPhotofront.total();
        dRvectran_dRvecDoubleside = dRvectran_dTvecDoubleside = 
        dTvectran_dRvecDoubleside = dTvectran_dTvecDoubleside =cv::Mat::zeros(3,3,CV_64F);
        Rvectran1 = RvecPhotofront;
        Tvectran1 = TvecPhotofront;
    }
    // std::cout <<"TvecCamera" << TvecCamera << std::endl;
    // std::cout <<"TvecPhoto" << TvecPhoto << std::endl;
  
    // std::cout <<"TvecPhotofront" << TvecPhotofront << std::endl;
    // std::cout <<"TvecDoubleside" << TvecDoubleside << std::endl;
    // std::cout <<"Tvectran" << Tvectran << std::endl;
    // std::cout <<"Tvectran1" << Tvectran1 << std::endl;
    
    
    // Mat dRvectran_dRvecDoubleside, dRvectran_dTvecDoubleside,
    //     dTvectran_dRvecDoubleside, dTvectran_dTvecDoubleside;
    // dRvectran_dRvecDoubleside =  dRvectran_dRvecPhotofront * dRvecPhotofront_dRvecDoubleside ;
    // dRvectran_dTvecDoubleside =  dRvectran_dTvecPhotofront * dTvecPhotofront_dTvecDoubleside ;
    // dTvectran_dRvecDoubleside =  dTvectran_dRvecPhotofront * dRvecPhotofront_dRvecDoubleside ;
    // dTvectran_dTvecDoubleside =  dTvectran_dTvecPhotofront * dTvecPhotofront_dTvecDoubleside ;  

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

    Mat imagePoints2, jacobian, dx_dRvecDoubleside, dx_dTvecDoubleside, dx_dRvecPhoto, dx_dTvecPhoto, E1;
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
    dx_dRvecDoubleside = jacobian.colRange(0, 3) * dRvectran_dRvecDoubleside + jacobian.colRange(3, 6) * dTvectran_dRvecDoubleside;
    dx_dTvecDoubleside = jacobian.colRange(0, 3) * dRvectran_dTvecDoubleside + jacobian.colRange(3, 6) * dTvectran_dTvecDoubleside;
    if (false && patternSide == BACK_PATTERN){
        std::cout<< "jacobian.colRange(0, 3):"<< jacobian.colRange(0, 3) << std::endl;
        std::cout<< "jacobian.colRange(3, 6):"<< jacobian.colRange(3, 6) << std::endl;
        std::cout<<"dRvectran_dRvecDoubleside:" << dRvectran_dRvecDoubleside<< std::endl;
        std::cout<<"dTvectran_dRvecDoubleside:" << dTvectran_dRvecDoubleside<< std::endl;
        std::cout<<"dRvectran_dTvecDoubleside:" << dRvectran_dTvecDoubleside<< std::endl;
        std::cout<<"dTvectran_dTvecDoubleside:" << dTvectran_dTvecDoubleside<< std::endl;
        std::cout<<"dx_dTvecDoubleside:" << dx_dTvecDoubleside<< std::endl;
        std::cout<<"dx_dRvecDoubleside:" << dx_dRvecDoubleside<< std::endl;
        int kkk=1;
    }


    dx_dRvecPhoto = jacobian.colRange(0, 3) * dRvectran_dRvecPhoto + jacobian.colRange(3, 6) * dTvectran_dRvecPhoto;
    dx_dTvecPhoto = jacobian.colRange(0, 3) * dRvectran_dTvecPhoto + jacobian.colRange(3, 6) * dTvectran_dTvecPhoto;

    jacobianDoubleside = cv::Mat(dx_dRvecDoubleside.rows, 6, CV_64F);
    jacobianPhoto = cv::Mat(dx_dRvecPhoto.rows, 6, CV_64F);

    dx_dRvecDoubleside.copyTo(jacobianDoubleside.colRange(0, 3));
    dx_dTvecDoubleside.copyTo(jacobianDoubleside.colRange(3, 6));
    dx_dRvecPhoto.copyTo(jacobianPhoto.colRange(0, 3));
    dx_dTvecPhoto.copyTo(jacobianPhoto.colRange(3, 6));

    cv::Mat tvecdiff = Tvectran1 - Tvectran;
    float norm1 = cv::norm(tvecdiff);
    if (norm1 > 10){
        int kk1 =0;
    }

}
int DoubleSideCalibration::getPhotoVertexParameters(int photoVertex){
    return photoVertex - _nCamera + 1;
}
void DoubleSideCalibration::computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE, Mat&deltaX)
{
    int nParam = (int)extrinsicParams.total();
    int nEdge = (int)_edgeList.size();
    std::vector<int> pointsLocation(nEdge+1, 0);
    std::cout << "extrinsicParams.colRange(3, 6)" << extrinsicParams.colRange(3, 6) << std::endl;
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int nPoints = (int)_objectPointsForEachCamera[_edgeList[edgeIdx].cameraVertex][_edgeList[edgeIdx].photoIndex].rows;
        pointsLocation[edgeIdx+1] = pointsLocation[edgeIdx] + nPoints*2;
    }

    JTJ_inv = Mat(nParam, nParam, CV_64F);
    JTE = Mat(nParam, 1, CV_64F);

    Mat J = Mat::zeros(pointsLocation[nEdge], nParam, CV_64F);
    Mat E = Mat::zeros(pointsLocation[nEdge], 1, CV_64F);
    Mat RvecDoubleSide = extrinsicParams.colRange(0, 0 + 3);
    Mat TvecDoubleSide = extrinsicParams.colRange(0 + 3, 0 + 6);

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


        int paraRow = getPhotoVertexParameters(photoVertex);
        Mat RvecPhoto = extrinsicParams.colRange(paraRow*6, paraRow*6 + 3);
        Mat TvecPhoto = extrinsicParams.colRange(paraRow*6 + 3, paraRow*6 + 6);
        assert(isValidPose(TvecPhoto));
        Mat RvecCamera, TvecCamera;
        // if (cameraVertex > 0)
        // {
            RvecCamera = camerasPose_rvec[cameraVertex];
            TvecCamera = camerasPose_tvec[cameraVertex];
        // }
        // else
        // {
        //     RvecCamera = Mat::zeros(3, 1, CV_32F);// 第一个相机姿态无旋转平移
        //     TvecCamera = Mat::zeros(3, 1, CV_32F);
        // }
        const float epsilon = 0.001;
        cv::Mat TvecDoubleSideEpsilon = TvecDoubleSide.clone();
        TvecDoubleSideEpsilon.at<float>(0) += epsilon;
        // std::cout << "TvecDoubleSideEpsilon:" << TvecDoubleSideEpsilon << std::endl;
        // std::cout << "TvecDoubleSide:" << TvecDoubleSide << std::endl;

        cv::Mat RvecDoubleSideEpsilon = RvecDoubleSide.clone();
        const int col = 0;
        RvecDoubleSideEpsilon.at<float>(col) += epsilon;
       

        Mat jacobianPhoto, jacobianDoubleside, error, error1, error2;
        // computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
        //     RvecDoubleSide, TvecDoubleSideEpsilon, Rvectran, Tvectran, 
        //     objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
        //     this->_xi[cameraVertex], jacobianPhoto, jacobianDoubleside, error2);        

        // computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
        //     RvecDoubleSideEpsilon, TvecDoubleSide, Rvectran, Tvectran, 
        //     objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
        //     this->_xi[cameraVertex], jacobianPhoto, jacobianDoubleside, error1);  

        computePhotoCameraJacobian(eg.patternSide, RvecPhoto, TvecPhoto, RvecCamera, TvecCamera, 
            RvecDoubleSide, TvecDoubleSide, Rvectran, Tvectran, 
            objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
            this->_xi[cameraVertex], jacobianPhoto, jacobianDoubleside, error);

        // if (eg.patternSide == BACK_PATTERN){
        //     cv::Mat errordiff = error1 - error;
        //     cv::Mat differenceRvec =errordiff / epsilon;
        //     cv::Mat differenceTvec = (error2 - error) / epsilon;
        //     cv::Mat derivative = jacobianDoubleside.col(col);
        //     // cv::Mat accuracy = difference - derivative;
        //     // std::cout << "RvecDoubleSideEpsilon:" << RvecDoubleSideEpsilon << std::endl;
        //     // std::cout << "RvecDoubleSide:" << RvecDoubleSide << std::endl;             
        //     // std::cout << "jacobianDoubleside" << jacobianDoubleside<< std::endl;
        //     std::cout << "differenceRvec" << differenceRvec.t() << std::endl;
        //     std::cout << "derivativeRvec" << derivative.t() << std::endl;
        //     std::cout << "differenceTvec" << differenceTvec.t() << std::endl;
        //     std::cout << "derivativeTvec" << jacobianDoubleside.col(3).t() << std::endl;
            
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
        jacobianDoubleside.copyTo(J.rowRange(rowBegin, rowEnd).
            colRange(0, 6));
        
        jacobianPhoto.copyTo(J.rowRange(rowBegin, rowEnd).
            colRange(paraRow*6, (paraRow + 1)*6));
        error.copyTo(E.rowRange(rowBegin, rowEnd));
        if (false && eg.patternSide == BACK_PATTERN){
            std::cout << "jacobianDoubleside:"<<jacobianDoubleside << std::endl;
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
    // std::cout<<"x:" << x.t() << std::endl;
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
void DoubleSideCalibration::writeDoubleSideTransform(){
    std::string dfilename = "doublesideTransform.yaml";
    cv::FileStorage fs = cv::FileStorage(dfilename, cv::FileStorage::WRITE);
    fs << "transform" << doubleSideTransform;
}
void DoubleSideCalibration::writeParameters(const std::string& filename) {
    // MyMultiCameraCalibration::writeParameters(filename);
    writeDoubleSideTransform();
}
void DoubleSideCalibration::vector2parameters(const Mat& parameters, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs)
{

    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.channels() == 1 && (int)parameters.total() == 6*(nVertex - _nCamera + 1));
    CV_Assert(parameters.depth() == CV_32F);
    parameters.reshape(1, 1);

    rvecVertex.reserve(0);
    tvecVertexs.resize(0);
    rvecVertex.push_back(Vec3f(parameters.colRange(0*6, 0*6 + 3)));
    tvecVertexs.push_back(Vec3f(parameters.colRange(0*6 + 3, 0*6 + 6)));    
    for (int i = _nCamera; i < nVertex ; ++i)
    {
        int row = getPhotoVertexParameters(i);
        rvecVertex.push_back(Vec3f(parameters.colRange(row*6, row*6 + 3)));
        tvecVertexs.push_back(Vec3f(parameters.colRange(row*6 + 3, row*6 + 6)));
        isValidPose(Vec3f(parameters.colRange(row*6 + 3, row*6 + 6)));
    }
    assert(rvecVertex.size() == nVertex - _nCamera + 1);
}
void DoubleSideCalibration::paras2vertex(const cv::Mat &extrinParam){
    std::vector<Vec3f> RvecVertex, TvecVertex;
    vector2parameters(extrinParam, RvecVertex, TvecVertex);
    Mat R;
    Mat pose = Mat::eye(4, 4, CV_32F);
    Rodrigues(RvecVertex[0], R);
    R.copyTo(pose.colRange(0, 3).rowRange(0, 3));
    Mat(TvecVertex[0]).reshape(1, 3).copyTo(pose.rowRange(0, 3).col(3));
    doubleSideTransform = pose;
    doubleSideTransform_rvec = RvecVertex[0];
    doubleSideTransform_tvec = TvecVertex[0];

    for (int verIdx = 1; verIdx < (int)_vertexList.size(); ++verIdx)
    {
        Mat R;
        Mat pose = Mat::eye(4, 4, CV_32F);
        Rodrigues(RvecVertex[verIdx + 1], R);
        R.copyTo(pose.colRange(0, 3).rowRange(0, 3));
        Mat(TvecVertex[verIdx + 1]).reshape(1, 3).copyTo(pose.rowRange(0, 3).col(3));
        _vertexList[verIdx].pose = pose;
		if (_verbose && verIdx < _nCamera)
		{
			std::cout << "final camera pose of camera " << verIdx << " is" << std::endl;
			std::cout << pose << std::endl;
		}
    }
}

double DoubleSideCalibration::computeProjectError(Mat& parameters)
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
    doubleSideTransform1 = Mat::eye(4, 4, CV_32F);
    transform = Mat::eye(4, 4, CV_32F);
    Mat Tdoubleside, Rdoubleside;
    Tdoubleside = Mat(TvecVertex[0]).reshape(1, 3);
    cv::Rodrigues(RvecVertex[0], Rdoubleside);   
    Rdoubleside.copyTo(doubleSideTransform1.rowRange(0, 3).colRange(0, 3));
    Tdoubleside.copyTo(doubleSideTransform1.rowRange(0, 3).col(3));
    //cameraPose * photoPose * doubleSideTransform = backpose
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat RPhoto, RCamera, TPhoto, TCamera;
        auto &edg = edgeList[edgeIdx];
        int cameraVertex = edg.cameraVertex;
        int photoVertex = edg.photoVertex;
        int PhotoIndex = edg.photoIndex;

        int row = getPhotoVertexParameters(photoVertex);
        TPhoto = Mat(TvecVertex[row]).reshape(1, 3);
        cv::Rodrigues(RvecVertex[row], RPhoto);
        RPhoto.copyTo(photoPose.rowRange(0, 3).colRange(0, 3));
        TPhoto.copyTo(photoPose.rowRange(0, 3).col(3));

        if (edg.patternSide == BACK_PATTERN){
            transform = camerasPose[cameraVertex] * photoPose * doubleSideTransform1;
        }else{
            transform = camerasPose[cameraVertex] * photoPose;
        }

        // if (cameraVertex == 0)
        // {
        //     RPhoto.copyTo(transform.rowRange(0, 3).colRange(0, 3));
        //     TPhoto.copyTo(transform.rowRange(0, 3).col(3));
        // }
        // else
        // {
        //     TCamera = Mat(TvecVertex[cameraVertex - 1]).reshape(1, 3);
        //     cv::Rodrigues(RvecVertex[cameraVertex - 1], RCamera);
        //     Mat(RCamera*RPhoto).copyTo(transform.rowRange(0, 3).colRange(0, 3));
        //     Mat(RCamera * TPhoto + TCamera).copyTo(transform.rowRange(0, 3).col(3));
        // }

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
}}