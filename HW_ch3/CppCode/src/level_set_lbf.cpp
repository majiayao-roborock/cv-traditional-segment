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

#include "level_set_lbf.h"
#include "display.h"
#include "level_set_cv.h"
#include "level_set_utils.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/core.hpp>

ParamLevelSetLBF::ParamLevelSetLBF(double forground_weight,
                                   double background_weight, double eps,
                                   double step_size, double length_term_weight,
                                   double gradient_term_weight, int window_size,
                                   double sigma)
    : ParamLevelSet(forground_weight, background_weight, eps, step_size,
                    length_term_weight, gradient_term_weight),
      window_size_(window_size),
      sigma_(sigma) {
}

LevelSetLBF::LevelSetLBF(cv::Mat image, const HeightMap& height_map,
                         const ParamLevelSetLBF& param)
    : GradientDescentBase(param.step_size_),
      phi_(image.rows, image.cols),
      last_phi_(phi_),
      param_(param),
      image_3_channel(image.clone()),
      image_64f_(image.size(), CV_64FC1),
      center_background_(0.0),
      center_foreground_(255.0),
      last_center_background_(0.0),
      last_center_foreground_(255.0),
      gauss_kernel_(get_gaussian_kernel(param.window_size_, param.sigma_)) {
    image.convertTo(image_64f_, CV_64FC1);
}

double LevelSetLBF::compute_energy() const {
    double result;
    // todo implement energy of LBF Model
    return result;
}
void LevelSetLBF::initialize() {
    // initialize lvl set :
    // height map already initilized in constructor

    // initilize centers :
    // centers already initalized in constructor
}

void LevelSetLBF::update() {
    visualize_lvl_set_segemenation(image_3_channel, phi_, 1);

    cv::Mat total_data_term_derivative =
        cv::Mat::zeros(image_64f_.size(), image_64f_.type());
    // todo implement the rest of update function of LBF Model
    cv::Mat update_step_length_term =
        param_.step_size_ * param_.length_term_weight_ *
        compute_derivative_length_term(phi_, param_.eps_);

    cv::Mat update_step_gradient_term = param_.step_size_ *
                                        param_.gradient_term_weight_ *
                                        compute_derivative_gradient_term(phi_);
    cv::Mat update_step = update_step_length_term + update_step_gradient_term;
    phi_.add(update_step);
}

void LevelSetLBF::update_center_in_window(int row, int col) {
    center_foreground_ =
        compute_center_in_window(row, col, param_.window_size_, gauss_kernel_,
                                 image_64f_, phi_, param_.eps_, false);
    std::cout << "center_foreground " << center_foreground_ << '\n';
    center_background_ =
        compute_center_in_window(row, col, param_.window_size_, gauss_kernel_,
                                 image_64f_, phi_, param_.eps_, true);
    std::cout << "center_background " << center_background_ << '\n';
}

cv::Mat LevelSetLBF::compute_data_term_derivative_in_window(int row,
                                                            int col) const {
    cv::Mat window_image =
        get_sub_image(image_64f_, row, col, param_.window_size_);
    cv::Mat e_foreground = compute_square_diff(
        image_64f_, center_foreground_ *
                        cv::Mat::ones(image_64f_.size(), image_64f_.type()));
    cv::Mat e_background = compute_square_diff(
        image_64f_, center_background_ *
                        cv::Mat::ones(image_64f_.size(), image_64f_.type()));
    return dirac(phi_).mul(param_.forground_weight_ * e_foreground -
                           param_.background_weight_ * e_background);
}

std::string LevelSetLBF::return_drive_class_name() const {
    return "Level Set LBF Model";
}

void LevelSetLBF::roll_back_state() {
    phi_ = last_phi_;
}
void LevelSetLBF::back_up_state() {
    last_phi_ = phi_;
}
void LevelSetLBF::print_terminate_info() const {
    std::cout << "Level set iteration finished." << std::endl;
}