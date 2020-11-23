/**
______________________________________________________________________
*********************************************************************
* @brief  This file is developed for the course of ShenLan XueYuan:
* Fundamental implementations of Computer Vision
* all rights preserved
* @author Xin Jin, Zhaoran Wu
* @contact: xinjin1109@gmail.com, zhaoran.wu1@gmail.com
*
______________________________________________________________________
*********************************************************************
**/

#include "gradient_descent_base.h"
#include "level_set_cv.h"

/**
 * @brief parameter of level set lbf model
 *
 * window_size_ : size of the local window
 * sigma_ : sigma to control the form of gaussian kernel
 *
 */
struct ParamLevelSetLBF : public ParamLevelSet {
    ParamLevelSetLBF(double forground_weight, double background_weight,
                     double eps, double step_size, double length_term_weight,
                     double gradient_term_weight, int window_size,
                     double sigma);
    int window_size_;
    double sigma_;
};

class LevelSetLBF : public GradientDescentBase {
   public:
    /**
     * @brief Construct a new Level Set LBF object
     *
     * @param image: a gray scale image in type of CV_8U
     * @param height_map: the function(hight map) phi
     * @param param: parameter of levelset lbf
     */
    LevelSetLBF(cv::Mat image, const HeightMap& height_map,
                const ParamLevelSetLBF& param);

   private:
    void update_center();
    void update_level_set();
    /**
     * @brief updae the window with center at position (col,row)
     */
    void update_center_in_window(int row, int col);

    cv::Mat compute_data_term_derivative_in_window(int row, int col) const;

    void update() override;
    void initialize() override;
    void roll_back_state() override;
    void back_up_state() override;
    void print_terminate_info() const override;
    double compute_energy() const override;
    std::string return_drive_class_name() const override;

   private:
    HeightMap phi_;
    HeightMap last_phi_;

    ParamLevelSetLBF param_;

    cv::Mat image_64f_;
    cv::Mat image_3_channel;

    double center_foreground_;
    double last_center_foreground_;

    double center_background_;
    double last_center_background_;

    cv::Mat gauss_kernel_;
};
