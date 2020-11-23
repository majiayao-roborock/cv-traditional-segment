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

#include "display.h"
#include "height_map.h"
#include "level_set_utils.h"
#include <opencv2/core.hpp>

int main(int argc, char** argv) {
    // define and and initialize a height map_map object
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    int rows = img.rows;
    int cols = img.cols;
    cv::Point2d center(cols / 2.f, rows / 2.f);
    double radius = std::min(rows, cols) / 4.f;
    HeightMap height_map(rows, cols, center, radius);
    // HeightMap height_map(rows, cols);

    cv::Mat height_map_draw = draw_height_map(height_map);
    cv::Mat height_map_with_contour =
        draw_points(height_map_draw, height_map.get_contour_points(),
                    cv::Scalar(255, 255, 255));
    disp_image(height_map_with_contour, "height map", 0);

    cv::Mat fore_back_ground = height_map.get_fore_background_label_map();
    disp_image(fore_back_ground, "fore- and background", 0);

    return 0;
}