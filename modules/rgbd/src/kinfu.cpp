// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "fast_icp.hpp"
#include "tsdf.hpp"
#include "kinfu_frame.hpp"

#include <vector>
 

namespace cv {
namespace kinfu {

Ptr<Params> Params::defaultParams()
{
    Params p;

    p.frameSize = Size(640, 480);

    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = p.frameSize.width/2 - 0.5f;
    cy = p.frameSize.height/2 - 0.5f;
    p.intr = Matx33f(fx,  0, cx,
                      0, fy, cy,
                      0,  0,  1);

    // 5000 for the 16-bit PNG files
    // 1 for the 32-bit float images in the ROS bag files
    p.depthFactor = 5000;

    // sigma_depth is scaled by depthFactor when calling bilateral filter
    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icpAngleThresh = (float)(30. * CV_PI / 180.); // radians
    p.icpDistThresh = 0.1f; // metersS

    p.icpIterations = {10, 5, 4};
    p.pyramidLevels = (int)p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    p.volumeDims = Vec3i::all(500); //number of voxels //<<---------------------------------------modification 
    float volSize = 3.f; //<<---------------------------------------modification 
    p.voxelSize = volSize/500.f; //meters //<<---------------------------------------modification 

    // default pose of volume cube
    // p.volumePose = Affine3f().translate(Vec3f(-volSize/2.f, -volSize/2.f, 0.5f));
    p.volumePose = Affine3f().translate(Vec3f(-1.5f, -1.4f,0.7f));//<<---------------------------------------modification 

    
    //------------------------------------------------------------------modification
    //we define the threshold of the map cube update here (in x and z direction)
    //x direction is move left or right && z direction is move front or back 
    p.map_update_threshold = 75; //consifer it as how many voxels
    //------------------------------------------------------------------modification


    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.25f;  //in voxel sizes
    // gradient delta factor is fixed at 1.0f and is not used
    //p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.lightPose = p.volume_pose.translation()/4; //meters
    p.lightPose = Vec3f::all(0.f); //meters

    // depth truncation is not used by default
    //p.icp_truncate_depth_dist = 0.f;        //meters, disabled

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    p->icpIterations = {5, 3, 2};
    p->pyramidLevels = (int)p->icpIterations.size();

    float volSize = 1.f; //<-----------------------------------modification
    p->volumeDims = Vec3i::all(128); //number of voxels
    p->voxelSize  = volSize/128.f;

    p->raycast_step_factor = 0.75f;  //in voxel sizes

    return p;
}

// T should be Mat or UMat
template< typename T >
class KinFuImpl : public KinFu
{
public:
    KinFuImpl(const Params& _params);
    virtual ~KinFuImpl();

    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const T& depth);

    bool move_more(const T& _depth);


    //----------------------------------------------------------------------------modification 
    void test_function();

    void test_function2();

    bool update_the_map (float x_direction_movement,float z_direction_movement);

    void modify_param(float x_in);

    void set_map_shift_with_camera();

    void set_last_update_position();


    //------------------------------------------------------------------------------modification
private:
    Params params;

    cv::Ptr<ICP> icp;
    cv::Ptr<TSDFVolume> volume;

    int frameCounter;
    Affine3f pose;
    std::vector<T> pyrPoints;
    std::vector<T> pyrNormals;

    //-------------------------------------------------------------------------------modification
    //float x_move_history; 
    
    //if the distance between current and last position in x or z direction is larger than the threshold, then
    //have to update the position in that direction 
    Vec3f last_update_position; 
    bool map_shift_with_camera;
};

//-----------------------------------------------------------------------------------modification
template< typename T >
void KinFuImpl<T>::test_function(){
    std::cout << "this is the test of the funciton creation" << std::endl;
    //volume->map_ignore_test();

    volume->front_test();
    // pose = Affine3f::Identity();
    
    // std::cout << "the threshold_voxel number is " << params.map_update_threshold << std::endl;
    
}

template< typename T >
void KinFuImpl<T>::test_function2(){
    std::cout << "2" << std::endl;
    volume->front_test2();
}

template< typename T >
void KinFuImpl<T>::modify_param(float x_in){
    Vec3f temp_translation = params.volumePose.translation();
    temp_translation[0]+=x_in;
    params.volumePose.translation() = temp_translation;
}

template< typename T >
void KinFuImpl<T>::set_map_shift_with_camera(){
    using namespace std;
    if (map_shift_with_camera == false){
        map_shift_with_camera = true;
        cout << "we set map_shift_with_camera to be true" << endl;
    }
    else{
        map_shift_with_camera = false;
        cout << "we set map_shift_with_camera to be false" << endl;
    }
}

template< typename T >
void KinFuImpl<T>::set_last_update_position(){
    last_update_position = pose.translation();
    std::cout << "we set the lasr_update_position" << std::endl;
}
 
//------------------------------------------------------------------------------------modification

template< typename T >
KinFuImpl<T>::KinFuImpl(const Params &_params) :
    params(_params),
    icp(makeICP(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    volume(makeTSDFVolume(params.volumeDims, params.voxelSize, params.volumePose,
                          params.tsdf_trunc_dist, params.tsdf_max_weight,
                          params.raycast_step_factor)),
    pyrPoints(), pyrNormals()
{
    reset();
    //---------------------------------------------------------------modification 
    last_update_position = {0,0,0};
    //---------------------------------------------------------------modification 

}

template< typename T >
void KinFuImpl<T>::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    volume->reset();
}

template< typename T >
KinFuImpl<T>::~KinFuImpl()
{ }

template< typename T >
const Params& KinFuImpl<T>::getParams() const
{
    return params;
}

template< typename T >
const Affine3f KinFuImpl<T>::getPose() const
{
    return pose;
}


template<>
bool KinFuImpl<Mat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    //std::cout << "the depth.dim is :" << _depth.size() << std::endl;//<<-----------modification 

    Mat depth;
    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getMat());
    }
}


template<>
bool KinFuImpl<UMat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    //std::cout << "it works all the time !" << std::endl;//<<-----------modification 

    UMat depth;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        //std::cout << "run first?" << std::endl;
        return updateT(depth);
    }
    else
    {
        //std::cout << "run second?" << std::endl;
        //std::cout << _depth.getUMat() << std::endl;
        return updateT(_depth.getUMat());
    }
}

 
//---------------------------------------------------------------------modification 
template< typename T >
bool KinFuImpl<T>::update_the_map (float x_direction_movement,float z_direction_movement){
    float threshold = params.map_update_threshold;
    float voxelSize = params.voxelSize;// meters how long is one voxel side
    float threshold_meter = threshold*voxelSize;
    //x_positive, x_begative, z_positive, z_negative
    std::vector<bool> dir_arr = {false,false,false,false};    
     
    //shift to right 
    if (std::abs(x_direction_movement) > threshold_meter && x_direction_movement >0 && map_shift_with_camera) {
        using namespace std;
        cout << "we need to update the map cube in x_position direction" << endl;
        pose = Affine3f::Identity();
        dir_arr[0] = true;
        volume->map_ignore_test(dir_arr,threshold);
        // last_update_position[0] = 0;
        return true;
    }
    
    //shift to left
    if (std::abs(x_direction_movement) > threshold_meter && x_direction_movement < 0 && map_shift_with_camera) {
        using namespace std;
        cout << "we need to update the map cube in x_negative direction" << endl;
        pose = Affine3f::Identity();
        dir_arr[1] = true;
        volume->map_ignore_test(dir_arr,threshold);
        return true;
    }

    //shift forward
    if ((std::abs(z_direction_movement) > threshold_meter ) && z_direction_movement>0 && map_shift_with_camera) {
        using namespace std;
        cout << "we need to update the map cube in z_positive direction" << endl;
        pose = Affine3f::Identity();
        dir_arr[2] = true;
        volume->map_ignore_test(dir_arr,threshold);
        return true;
    }

    //shift back 
    if (std::abs(z_direction_movement) > threshold_meter && z_direction_movement<0 && map_shift_with_camera) {
        using namespace std;
        cout << "we need to update the map cube in z_negative direction" << endl;
        pose = Affine3f::Identity();
        dir_arr[3] = true;
        volume->map_ignore_test(dir_arr,threshold);
        return true;
    }
    return false;
}
//---------------------------------------------------------------------modification 


template< typename T >
bool KinFuImpl<T>::updateT(const T& _depth)
{
    CV_TRACE_FUNCTION();

    T depth;
    if(_depth.type() != DEPTH_TYPE){
        _depth.convertTo(depth, DEPTH_TYPE);
        //std::cout << "do you inter this if statement?" << std::endl;
    }
    else{
        depth = _depth;
    }
    std::vector<T> newPoints, newNormals;
    makeFrameFromDepth(depth, newPoints, newNormals, params.intr,
                       params.pyramidLevels,
                       params.depthFactor,
                       params.bilateral_sigma_depth,
                       params.bilateral_sigma_spatial,
                       params.bilateral_kernel_size);



//-----------------------------------------------------------------------------
    //-----------converte two vector to mat and use imshow to visualize them-----------
    //1. newPoints:

    // using namespace std;
    // cout <<"the size of newNormals is " << newNormals.size() << endl;

    // T umat11 = newPoints[0];
    // if (typeid(umat11) == typeid(cv::UMat)){
    //     cout << "it is mat" << endl;
    // }
    // // Mat mat1;
    // // umat11.copyTo(mat1);
    // cv::imshow("umat11",umat11);

    // T umat1 = newNormals[0];
    // // Mat mat1;
    // // umat1.copyTo(mat1);
    // cv::imshow("umat1",umat1);


    // T umat22 = newPoints[1];
    // // Mat mat2;
    // // umat2.copyTo(mat2);
    // cv::imshow("umat22",umat22);

    //  T umat2 = newNormals[1];
    // // Mat mat2;
    // // umat2.copyTo(mat2);
    // cv::imshow("umat2",umat2);

    // T umat33 = newPoints[2];
    // // Mat mat3;
    // // umat3.copyTo(mat3);
    // cv::imshow("umat33",umat33);

    // T umat3 = newNormals[2];
    // // Mat mat3;
    // // umat3.copyTo(mat3);
    // cv::imshow("umat3",umat3);

    // T umat4 = newPoints[3];
    // Mat mat4;
    // umat4.copyTo(mat4);
    // cv::imshow("umat4",mat4);
    // cv::Mat img1(newPoints);
    // std::cout<< "the size is " << newNormals.size() << std::cout; 
    // cv::imshow("newPoints",img1);
    // cv::waitKey(0);
    //std::cout << "the frame counter is: " << frameCounter << std::endl; 
    //--------------------------------------------------------------------------------
    if(frameCounter == 0)
    {
        // use depth instead of distance
        std::cout << "in z direction_first_frame  " << pose.translation()[2] << std::endl;
        volume->integrate(depth, params.depthFactor, pose, params.intr);

        // cv::imshow("the new depth",depth);
        pyrPoints  = newPoints;
        pyrNormals = newNormals;

    }
    else
    {   
        //--------------------------------------------------------------------------modification 
        std::cout <<  "in z direction " << pose.translation()[2];
        std::cout <<  "in x direction " << /*x_move_history - */pose.translation()[0] << std::endl;
        // std::cout << "the z direction last store: " << last_update_position[2]<< " " 
        //     << "the x direction last store: " << last_update_position[0] << "     "
        //     << pose.translation()[2] - last_update_position[2] << ", "
        //     << pose.translation()[0] - last_update_position[0]  << std::endl;

        //calcuate the difference in x and z direction 
        float x_shift = pose.translation()[0] - last_update_position[0];
        float z_shift = pose.translation()[2] - last_update_position[2];
        //update the last_update_position 
        // float threshold_meter = params.map_update_threshold * params.voxelSize;
        // if (std::abs(x_shift)>threshold_meter){
        //     last_update_position[0] += x_shift;
        //     std::cout <<"update the last_update_position in x direction " << std::endl;
        // }
        // if (std::abs(z_shift)>threshold_meter){
        //     last_update_position[2] += z_shift;
        //     std::cout <<"update the last_update_position in z direction " << std::endl;

        // }

        // std::cout << "z_shift: " << z_shift << ", " << "x_shift: " << x_shift << std::endl;
        //use the shift distance to check if 
        if(update_the_map(x_shift, z_shift)){
            std::cout << "we let the x go back to zero" << std::endl;
            // set_last_update_position();
            // frameCounter = -1;
            //return false;
        }

        // if(update_the_map(pose.translation()[0], pose.translation()[2])){
        //     std::cout << "we let the x go back to zero" << std::endl;
        //     // frameCounter = -1;
        //     //return false;
        // }

        //---------------------------------------------------------------------------modification
        


        Affine3f affine;

        // Vec3f translation_temp1 = affine.translation();
        // std::cout << "the translation is: " << translation_temp1 << std::endl;
    

        bool success = icp->estimateTransform(affine, pyrPoints, pyrNormals, newPoints, newNormals);
        // Vec3f translation_temp2 = affine.translation();
        // std::cout << "the new translation is: " << translation_temp2 << std::endl;

        // using namespace std;
        // cout << "the size of pyrPoints is " << pyrPoints[0].size()<< ", "<< pyrPoints[1].size() <<", " << pyrPoints[2].size() <<", "<< endl;
        // if(!success){
        //     std::cout << "--------------------------------!!!--------------------------------" << std::endl;
        //     return false;
        // }
        using namespace std;

        float x_position = pose.translation()[0];//-------------------------modification 
        float z_position = pose.translation()[2];//-------------------------modification 
        // cout << "the old one is " << z_position << endl;

        pose = pose * affine;
        // pose = Affine3f::Identity();
        //------------------------------------------------------------------------------modification-----ignore the noise
        float new_x_posisition = pose.translation()[0];
        float new_z_posisition = pose.translation()[2];
        // cout << "the new one is " << new_z_posisition << endl;
        if (std::abs(new_x_posisition-x_position)<0.005f){
            Vec3f new_translation = pose.translation();
            new_translation[0] = x_position;
            pose.translation(new_translation);
        }

        if (std::abs(new_z_posisition-z_position)<0.005f){
            Vec3f new_translation = pose.translation();
            new_translation[2] = z_position;
            pose.translation(new_translation);
        }
        // new_z_posisition = pose.translation()[2];
        // cout << "the new one is " << new_z_posisition << endl;
        //------------------------------------------------------------------------------modification 
        
        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());

        // std::cout << "how much is the camera translation is " << tnorm * 100 << " cm" << std::endl;
        // std::cout << "how much is the camera rotation is " << rnorm << " degree" << std::endl;

        // std::cout << "x coordinate-------------" << affine.translation()[0] << std::endl;

        // if (tnorm < 0.001){
        //     tnorm = 0.0;
        // }
        // movement_total=tnorm;
        // std::cout << "total distance" << movement_total << "m" << std::endl;
        // if (movement_total > 5){
        //     movement_total = 0;
        //     return false;
        // }
        // else{
        //     return true;
        // }
        Vec8i size = volume->neighbourCoords;
       



        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume->integrate(depth, params.depthFactor, pose, params.intr);
        }




        T& points  = pyrPoints [0];
        T& normals = pyrNormals[0];
        volume->raycast(pose, params.intr, params.frameSize, points, normals);
        // build a pyramid of points and normals
        buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals,
                                  params.pyramidLevels);
        //std::cout << "do nothing here " << std::endl;
    }

    frameCounter++;
    return true;
}

///////////////////////////////////////////////////////////////





template< typename T >
void KinFuImpl<T>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);

    const Affine3f id = Affine3f::Identity();
    if((cameraPose.rotation() == pose.rotation() && cameraPose.translation() == pose.translation()) ||
       (cameraPose.rotation() == id.rotation()   && cameraPose.translation() == id.translation()))
    {
        renderPointsNormals(pyrPoints[0], pyrNormals[0], image, params.lightPose);
    }
    else
    {
        T points, normals;
        volume->raycast(cameraPose, params.intr, params.frameSize, points, normals);
        renderPointsNormals(points, normals, image, params.lightPose);
    }
}


template< typename T >
void KinFuImpl<T>::getCloud(OutputArray p, OutputArray n) const
{
    volume->fetchPointsNormals(p, n);
}


template< typename T >
void KinFuImpl<T>::getPoints(OutputArray points) const
{
    volume->fetchPointsNormals(points, noArray());
}


template< typename T >
void KinFuImpl<T>::getNormals(InputArray points, OutputArray normals) const
{
    volume->fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<KinFu> KinFu::create(const Ptr<Params>& params)
{
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL())
        return makePtr< KinFuImpl<UMat> >(*params);
#endif
    return makePtr< KinFuImpl<Mat> >(*params);
}

#else
Ptr<KinFu> KinFu::create(const Ptr<Params>& /*params*/)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

KinFu::~KinFu() {}

} // namespace kinfu
} // namespace cv
