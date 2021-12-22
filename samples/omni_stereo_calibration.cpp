#include "omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;

const char * usage =
	"\n example command line for calibrate a pair of omnidirectional camera.\n"
	"    omni_stereo_calibration -w 8 -h 6 -sw 2.4399 -sh 2.4399 imagelist_left.xml imagelist_right.xml\n"
	" \n"
	" the file image_list_1.xml and image_list_2.xml generated by imagelist_creator as\n"
	"imagelist_creator image_list_1.xml *.*";

static void help()
{
	printf("\n This is a sample for omnidirectional camera calibration.\n"
		"Usage: omni_calibration\n"
		"    -w <board_width>    # the number of inner corners per one of board dimension\n"
		"    -h <board_height>    # the number of inner corners per another board dimension\n"
		"    [-sw <square_width>] # the width of square in some user-defined units (1 by default)\n"
		"    [-sh <square_height>] # the height of square in some user-defined units (1 by default)\n"
		"    [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
		"    [-fs <fix_skew>] # fix skew\n"
		"    [-fp ] # fix the principal point at the center\n"
		"    input_data_1 # input data - text file with a list of the images of the first camera, which is generated by imagelist_creator"
        "    input_data_2 # input data - text file with a list of the images of the second camera, which is generated by imagelist_creator"
		);
	printf("\n %s", usage);
}

static void calcChessboardCorners(Size boardSize, double square_width, double square_height,
    Mat& corners)
{
    // corners has type of CV_64FC3
    corners.release();
    int n = boardSize.width * boardSize.height;
    corners.create(n, 1, CV_64FC3);
    Vec3d *ptr = corners.ptr<Vec3d>();
    for (int i = 0; i < boardSize.height; ++i)
    {
        for (int j = 0; j < boardSize.width; ++j)
        {
            ptr[i*boardSize.width + j] = Vec3d(double(j * square_width), double(i * square_height), 0.0);
        }
    }
}

static bool detecChessboardCorners(const vector<string>& list1, vector<string>& list_detected_1,
    const vector<string>& list2, vector<string>& list_detected_2,
    vector<Mat>& image_points_1, vector<Mat>& image_points_2, Size boardSize, Size& imageSize1, Size& imageSize2)
{
    image_points_1.resize(0);
    image_points_2.resize(0);
    list_detected_1.resize(0);
    list_detected_2.resize(0);
    int n_img = (int)list1.size();
    Mat img_l, img_r;
    for(int i = 0; i < n_img; ++i)
    {
        Mat points_1, points_2;
        img_l = imread(list1[i], IMREAD_GRAYSCALE);
        img_r = imread(list2[i], IMREAD_GRAYSCALE);
        bool found_l = findChessboardCorners( img_l, boardSize, points_1);
        bool found_r = findChessboardCorners( img_r, boardSize, points_2);
        if (found_l && found_r)
        {
            if (points_1.type() != CV_64FC2)
                points_1.convertTo(points_1, CV_64FC2);
            if (points_2.type() != CV_64FC2)
                points_2.convertTo(points_2, CV_64FC2);
            image_points_1.push_back(points_1);
            image_points_2.push_back(points_2);
            list_detected_1.push_back(list1[i]);
            list_detected_2.push_back(list2[i]);
        }
    }
    if (!img_l.empty())
        imageSize1 = img_l.size();
    if (!img_r.empty())
    {
        imageSize2 = img_r.size();
    }

    if (image_points_1.size() < 3)
        return false;
    else
        return true;
}

static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

static void saveCameraParams( const string & filename, const int flags, const Mat& cameraMatrix1, const Mat& cameraMatrix2, const Mat& distCoeffs1,
    const Mat& disCoeffs2, const double xi1, const double xi2, const Vec3d rvec, const Vec3d tvec,
    const vector<Vec3d>& rvecs, const vector<Vec3d>& tvecs, vector<string> detec_list_1, vector<string> detec_list_2,
    const Mat& idx, const double rms, const vector<Mat>& imagePoints1, const vector<Mat>& imagePoints2)
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if ( !rvecs.empty())
        fs << "nFrames" << (int)rvecs.size();

    if ( flags != 0)
    {
        sprintf( buf, "flags: %s%s%s%s%s%s%s%s%s",
            flags & omnidir::CALIB_USE_GUESS ? "+use_intrinsic_guess" : "",
            flags & omnidir::CALIB_FIX_SKEW ? "+fix_skew" : "",
            flags & omnidir::CALIB_FIX_K1 ? "+fix_k1" : "",
            flags & omnidir::CALIB_FIX_K2 ? "+fix_k2" : "",
            flags & omnidir::CALIB_FIX_P1 ? "+fix_p1" : "",
            flags & omnidir::CALIB_FIX_P2 ? "+fix_p2" : "",
            flags & omnidir::CALIB_FIX_XI ? "+fix_xi" : "",
            flags & omnidir::CALIB_FIX_GAMMA ? "+fix_gamma" : "",
            flags & omnidir::CALIB_FIX_CENTER ? "+fix_center" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix_1" << cameraMatrix1;
    fs << "distortion_coefficients_1" << distCoeffs1;
    fs << "xi_1" << xi1;

    fs << "camera_matrix_2" << cameraMatrix2;
    fs << "distortion_coefficients_2" << disCoeffs2;
    fs << "xi_2" << xi2;

    Mat om_t(1, 6, CV_64F);
    Mat(rvec).reshape(1, 1).copyTo(om_t.colRange(0, 3));
    Mat(tvec).reshape(1, 1).copyTo(om_t.colRange(3, 6));
    //cvWriteComment( *fs, "6-tuples (rotation vector + translation vector) for each view", 0 );
    fs << "extrinsic_parameters" << om_t;

    if ( !rvecs.empty() && !tvecs.empty() )
    {
        Mat rvec_tvec((int)rvecs.size(), 6, CV_64F);
        for (int i = 0; i < (int)rvecs.size(); ++i)
        {
            Mat(rvecs[i]).reshape(1, 1).copyTo(rvec_tvec(Rect(0, i, 3, 1)));
            Mat(tvecs[i]).reshape(1, 1).copyTo(rvec_tvec(Rect(3, i, 3, 1)));
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters_1" << rvec_tvec;
    }

	fs << "rms" << rms;

    //cvWriteComment( *fs, "names of images that are acturally used in calibration", 0 );
    fs << "used_imgs_1" << "[";
    for (int i = 0;  i < (int)idx.total(); ++i)
    {
        fs << detec_list_1[(int)idx.at<int>(i)];
    }
    fs << "]";

    fs << "used_imgs_2" << "[";
    for (int i = 0;  i < (int)idx.total(); ++i)
    {
        fs << detec_list_2[(int)idx.at<int>(i)];
    }
    fs << "]";

    if ( !imagePoints1.empty() )
    {
        Mat imageMat((int)imagePoints1.size(), (int)imagePoints1[0].total(), CV_64FC2);
        for (int i = 0; i < (int)imagePoints1.size(); ++i)
        {
            Mat r = imageMat.row(i).reshape(2, imageMat.cols);
            Mat imagei(imagePoints1[i]);
            imagei.copyTo(r);
        }
        fs << "image_points_1" << imageMat;
    }

    if ( !imagePoints2.empty() )
    {
        Mat imageMat((int)imagePoints2.size(), (int)imagePoints2[0].total(), CV_64FC2);
        for (int i = 0; i < (int)imagePoints2.size(); ++i)
        {
            Mat r = imageMat.row(i).reshape(2, imageMat.cols);
            Mat imagei(imagePoints2[i]);
            imagei.copyTo(r);
        }
        fs << "image_points_2" << imageMat;
    }
}

int main(int argc, char** argv)
{
    Size boardSize, imageSize1, imageSize2;
    int flags = 0;
    double square_width = 0.0, square_height = 0.0;
    const char* outputFilename = "out_camera_params_stereo.xml";
    const char* inputFilename1 = 0;
    const char* inputFilename2 = 0;
    vector<Mat> objectPoints;
    vector<Mat> imagePoints1;
    vector<Mat> imagePoints2;

    if(argc < 2)
    {
        help();
        return 1;
    }

    bool fist_flag = true;
    for(int i = 1; i < argc; i++)
    {
        const char* s = argv[i];
        if( strcmp( s, "-w") == 0)
        {
            if( sscanf( argv[++i], "%u", &boardSize.width ) != 1 || boardSize.width <= 0 )
                return fprintf( stderr, "Invalid board width\n" ), -1;
        }
        else if( strcmp( s, "-h" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &boardSize.height ) != 1 || boardSize.height <= 0 )
                return fprintf( stderr, "Invalid board height\n" ), -1;
        }
        else if( strcmp( s, "-sw" ) == 0 )
        {
            if( sscanf( argv[++i], "%lf", &square_width ) != 1 || square_width <= 0 )
                return fprintf(stderr, "Invalid square width\n"), -1;
        }
        else if( strcmp( s, "-sh" ) == 0 )
        {
            if( sscanf( argv[++i], "%lf", &square_height) != 1 || square_height <= 0 )
                return fprintf(stderr, "Invalid square height\n"), -1;
        }
        else if( strcmp( s, "-o" ) == 0 )
        {
            outputFilename = argv[++i];
        }
        else if( strcmp( s, "-fs" ) == 0 )
        {
            flags |= omnidir::CALIB_FIX_SKEW;
        }
        else if( strcmp( s, "-fp" ) == 0 )
        {
            flags |= omnidir::CALIB_FIX_CENTER;
        }
        else if( s[0] != '-' && fist_flag)
        {
            fist_flag = false;
            inputFilename1 = s;
        }
        else if( s[0] != '-' && !fist_flag)
        {
            inputFilename2 = s;
        }
        else
        {
            return fprintf( stderr, "Unknown option %s\n", s ), -1;
        }
    }

    // get image name list
    vector<string> image_list1, detec_list_1, image_list2, detec_list_2;
    if((!readStringList(inputFilename1, image_list1)) || (!readStringList(inputFilename2, image_list2)))
        return fprintf( stderr, "Failed to read image list\n"), -1;

    // find corners in images
    // some images may be failed in automatic corner detection, passed cases are in detec_list
    if(!detecChessboardCorners(image_list1, detec_list_1, image_list2, detec_list_2,
        imagePoints1, imagePoints2, boardSize, imageSize1, imageSize2))
        return fprintf(stderr, "Not enough corner detected images\n"), -1;

    // calculate object coordinates
    Mat object;
    calcChessboardCorners(boardSize, square_width, square_height, object);
    for(int i = 0; i < (int)detec_list_1.size(); ++i)
    {
        objectPoints.push_back(object);
    }

    // run calibration, some images are discarded in calibration process because they are failed
    // in initialization. Retained image indexes are in idx variable.
    Mat K1, K2, D1, D2, xi1, xi2, idx;
    vector<Vec3d> rvecs, tvecs;
    Vec3d rvec, tvec;
    double _xi1, _xi2, rms;
    TermCriteria criteria(3, 200, 1e-8);
    rms = omnidir::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1,
        K2, xi2, D2, rvec, tvec, rvecs, tvecs, flags, criteria, idx);
    _xi1 = xi1.at<double>(0);
    _xi2 = xi2.at<double>(0);

    saveCameraParams(outputFilename, flags, K1, K2, D1, D2, _xi1, _xi2, rvec, tvec, rvecs, tvecs,
		detec_list_1, detec_list_2, idx, rms, imagePoints1, imagePoints2);
}
