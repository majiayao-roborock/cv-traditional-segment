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
#include "level_set_lbf.h"
#include "level_set_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

int main(int argc, char** argv) {
    // define and and initialize a height_map object
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    // cv::GaussianBlur(img, img, cv::Size(3, 3), 3);
    HeightMap height_map(img.rows, img.cols);
    ParamLevelSetLBF param_level_set_lbf(2e-3, 2e-3, 2.5, 1e-3, 40, 1.2, 21,
                                         11);

    cv::Mat dx = do_sobel(img, 1);

    LevelSetLBF level_set_lbf(img, height_map, param_level_set_lbf);
    level_set_lbf.run(1e4);

    return 0;
}