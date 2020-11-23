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

#pragma once
#include "gradient_descent_base.h"
#include "height_map.h"
#include "level_set_utils.h"

/**
 * @brief parameter of level set
 *
 *
 *   forground_weight_ : weight of foregroud
 *   background_weight_: weight of foregroud
 *   eps_: epsilon of heaviside function
 *   step_size_: step size factor of gradient decent
 *   length_term_weight_: weight of length term
 *   gradient_term_weight_: weight of gradient term
 *
 */
struct ParamLevelSet {
    ParamLevelSet(double forground_weight, double background_weight, double eps,
                  double step_size, double length_term_weight,
                  double gradient_term_weight);
    double forground_weight_;
    double background_weight_;
    double eps_;
    double step_size_;
    double length_term_weight_;
    double gradient_term_weight_;
};

/**
 * @brief class LevelSetCV : Chan Verse model of Level Set
 *
 */
class LevelSetCV : public GradientDescentBase {
   public:
    /**
     * @brief Construct a new Level Set CV object
     *
     * @param image: a gray scale image in type of CV_8U
     * @param height_map: the function(hight map) phi
     * @param param: parameter of levelset cv
     */
    LevelSetCV(cv::Mat image, const HeightMap& height_map,
               const ParamLevelSet& param);

   private:
    /**
     * @brief update the phi function
     *
     */
    void update_level_set();
    /**
     * @brief update the gray value center of foreground and background
     *
     */
    void update_center();

    void update() override;
    void initialize() override;
    void roll_back_state() override;
    void back_up_state() override;
    void print_terminate_info() const override;
    double compute_energy() const override;
    std::string return_drive_class_name() const override;

    HeightMap phi_;
    HeightMap last_phi_;

    ParamLevelSet param_;

    cv::Mat image_64f_;       // original image in the type of CV_64F
    cv::Mat image_3_channel;  // original image in gray scale but with 3 channel

    double center_foreground_;
    double last_center_foreground_;

    double center_background_;
    double last_center_background_;
};
