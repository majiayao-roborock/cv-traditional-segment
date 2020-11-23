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
#include <opencv2/core/types.hpp>

int main(int argc, char** argv) {
    // define and and initialize a height_map object
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::GaussianBlur(img, img, cv::Size(3, 3), 3);
    ParamLevelSet param_level_set_cv(5e-2, 5e-2, 1.0, 2e-2, 20, 1.2);

    HeightMap height_map(img.rows, img.cols);
    LevelSetCV level_set_cv(img, height_map, param_level_set_cv);
    level_set_cv.run(1e4);

    return 0;
}