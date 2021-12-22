
#include "mymulticalib.hpp"

#include "doubleSide.hpp"
using namespace std;
using namespace cv;

const char * usage =
"\n example command line for multi-camera calibration by using random pattern \n"
"   multi_cameras_calibration -nc 5 -pw 800 -ph 600 -ct 1 -fe 0 -nm 25 -v 0 multi_camera_omnidir.xml \n"
"\n"
" the file multi_camera_omnidir.xml is generated by imagelist_creator as \n"
" imagelist_creator multi_camera_omnidir.xml *.* \n"
" note the first filename in multi_camera_omnidir.xml is the pattern, the rest are photo names,\n"
" photo names should be in form of cameraIdx-timestamp.*, and cameraIdx starts from 0";

static void help()
{
    printf("\n This is a sample for multi-camera calibration, so far it only support random pattern,\n"
           "see randomPattern.hpp for detail. Pinhole and omnidirectional cameras are both supported, \n"
           "for omnidirectional camera, see omnidir.hpp for detail.\n"
           "Usage: mutiCamCalib \n"
           "    -nc <num_camera> # number of cameras \n"
           "    -pw <pattern_width> # physical width of random pattern \n"
           "    -ph <pattern_height> # physical height of random pattern \n"
           "    -ct <camera_type> # camera type, 0 for pinhole and 1 for omnidirectional \n"
           "    -fe # whether show feature extraction\n"
           "    -nm # number of minimal matches of an image \n"
		   "	-v # whether show verbose information \n"
           "    input_data # text file with pattern file names and a list of photo names, the file is generated by imagelist_creator \n");
    printf("\n %s", usage);
}


int main(int argc, char** argv)
{
    float patternWidth = 0.0f, patternHeight = 0.0f;
    int nMiniMatches = 0, cameraType = 0;
    const char* outputFilename = "multi-camera-results.xml";
    const char* inputFilename = 0;
    int showFeatureExtraction = 0, verbose = 0;



    // do multi-camera calibration
    cv::Size patternFront = cv::Size(8,11);
    cv::Size patternBack = cv::Size(7,10);

#if 1
    std::string a_dataFolder =  "/2t/data/recordedSamples/board/12.14opsite/color";
    const std::string a_cameraConfigFolder = "/home/dd/working/pypose/configs/warmup1hour";
    const std::string a_DoubleSideConfig = "/home/dd/working/pypose/configs/doublesideTransform.yaml";
    std::vector<std::string> a_cameraSerials = {"839112060578", "839512061262","f0220380"};
    int nCameras = a_cameraSerials.size();    
    cv::multicalib::MyMultiCameraCalibration multiCalib(a_cameraSerials, cameraType, nCameras, 
        a_dataFolder, a_cameraConfigFolder, a_DoubleSideConfig, patternFront, patternBack,
        patternWidth, patternHeight, verbose, showFeatureExtraction, nMiniMatches);


#else
    std::string a_dataFolder =  "/2t/data/recordedSamples/board/12.14opsite/color";
    const std::string a_cameraConfigFolder = "/home/dd/working/pypose/configs/warmup1hour";

    std::vector<std::string> a_cameraSerials = {"839112060578","f0220380"};
    int nCameras = a_cameraSerials.size();    
    cv::multicalib::DoubleSideCalibration multiCalib(a_cameraSerials, cameraType, nCameras, 
        a_dataFolder, a_cameraConfigFolder, patternFront, patternBack,
        patternWidth, patternHeight, verbose, showFeatureExtraction, nMiniMatches);
#endif

    multiCalib.loadImages();
    multiCalib.initialize();
    multiCalib.optimizeExtrinsics();
    std::set<std::string> outliers = multiCalib.removeOutlier();
    std::cout <<"number of outliers: " << outliers.size() << std::endl;
    multiCalib.reset();
    multiCalib.loadImages(outliers);
    multiCalib.initialize();
    multiCalib.optimizeExtrinsics();    
    // the above three lines can be replaced by multiCalib.run();


	multiCalib.writeParameters(outputFilename);
}