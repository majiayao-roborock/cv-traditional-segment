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

#include "height_map.h"
#include "display.h"
#include "level_set_utils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

/**
 * @brief draw sdf map for visualization
 *
 * @param sdf_map to be visulized
 * @return cv::Mat the visualzation image
 */
cv::Mat draw_height_map(const HeightMap& height_map) {
    assert(!height_map.get_map().empty());
    return apply_jetmap(height_map.get_map());
}

HeightMap::HeightMap(int rows, int cols, cv::Point center, double radius)
    : map_(cv::Mat::zeros(cv::Size(cols, rows), CV_64F)) {
    // todo implement the sign distance function, which background is plus sign,
    // todo  foreground is minus sign
}

HeightMap::HeightMap(int rows, int cols)
    : map_(2 * cv::Mat::ones(cv::Size(cols, rows), CV_64F)) {
    double percentage = 0.2;
    map_(cv::Rect2d(
        cv::Point(round(percentage * cols), round(percentage * rows)),
        cv::Point(cols - round(percentage * cols),
                  rows - round(percentage * rows)))) =
        -2 * cv::Mat::ones(cv::Size(cols - round(percentage * cols) * 2,
                                    rows - round(percentage * rows) * 2),
                           CV_64F);
}

cv::Mat HeightMap::get_fore_background_label_map() const {
    cv::Mat fore_background = map_.clone();
    cv::threshold(map_, fore_background, 0, 255, cv::THRESH_BINARY_INV);
    return fore_background;
}

double HeightMap::get_gradient_magnitude_level_set() {
    cv::Mat map_dev_x = do_sobel(map_, 0);
    cv::Mat map_dev_y = do_sobel(map_, 1);
    cv::Mat mag_grad_map;
    cv::sqrt(map_dev_x.mul(map_dev_x) + map_dev_y.mul(map_dev_y), mag_grad_map);
    return 0.5 * (mag_grad_map - 1.0).dot(mag_grad_map - 1.0);
}

void HeightMap::add(cv::Mat step) {
    map_ += step;
}

cv::Mat HeightMap::get_contour_points() const {
    std::vector<cv::Vec2d> contour_vec;

    // todo get the contour points on the zero level set
    // Hints: if the point is contour, put it in the contour_vec

    cv::Mat contour(cv::Size(2, contour_vec.size()), CV_64FC1);
    for (int i = 0; i < contour_vec.size(); i++) {
        contour.at<cv::Vec2d>(i) = contour_vec[i];
    }

    return contour;
}

cv::Mat HeightMap::get_map() const {
    return map_.clone();
}