
#include "display.h"

void draw_optical_flow(cv::Mat& fx, cv::Mat& fy, cv::Mat& cflowmap, int step,
                       double scaleFactor, cv::Scalar& color) {
    for (int r = 0; r < cflowmap.rows; r += step)
        for (int c = 0; c < cflowmap.cols; c += step) {
            cv::Point2f fxy;

            fxy.x = fx.at<double>(r, c);
            fxy.y = fy.at<double>(r, c);

            if (fxy.x != 0 || fxy.y != 0) {
                cv::line(cflowmap, cv::Point(c, r),
                         cv::Point(cvRound(c + (fxy.x) * scaleFactor),
                                   cvRound(r + (fxy.y) * scaleFactor)),
                         color, 1, cv::LINE_AA);
            }
            cv::circle(cflowmap, cv::Point(c, r), 1, cv::Scalar(255, 0, 0), 1);
        }
}

void display_gvf(cv::Mat fx, cv::Mat fy, int delay, bool save = false) {
    cv::Mat cflowmap = cv::Mat::zeros(fx.size(), CV_8UC3);

    int step = 8;
    double scaleFactor = 7;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Mat disp_fx = fx.clone();
    cv::Mat disp_fy = fy.clone();
    cv::normalize(disp_fx, disp_fx, -1, 1, cv::NORM_MINMAX);
    cv::normalize(disp_fy, disp_fy, -1, 1, cv::NORM_MINMAX);
    draw_optical_flow(disp_fx, disp_fy, cflowmap, step, scaleFactor, color);
    disp_image(cflowmap, "gvf display", delay);
    if (save) cv::imwrite("gvf_display.png", cflowmap);
}

//--Overloaded functions to display an image in a new window--//
void disp_image(cv::Mat& img) {
    if (img.empty()) {  // Read image and display after checking for image
                        // validity
        std::cout << "Error reading image file!";
        std::cin.ignore();
    } else {
        cv::namedWindow("Image", 0);
        cv::imshow("Image", img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName) {
    if (img.empty()) {  // Read image and display after checking for image
                        // validity
        std::cout << "Error reading image File!";
        std::cin.ignore();
    } else {
        cv::imshow(windowName, img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName, int delay) {
    if (img.empty()) {  // Read image and display after checking for image
                        // validity
        std::cout << "Error reading image File!";
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, img);
        cv::waitKey(delay);
    }
}

void disp_image(cv::Mat& img, cv::String windowName, cv::String error_msg) {
    if (img.empty()) {  // Read image and display after checking for image
                        // validity
        std::cout << error_msg;
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName, cv::String error_msg,
                int delay) {
    if (img.empty()) {  // Read image and display after checking for image
                        // validity
        std::cout << error_msg;
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, 0);
        cv::imshow(windowName, img);
        cv::waitKey(delay);
    }
}

/**
 * @brief apply a jetmap to an image.
 * @param image : image to be applied with jet map
 * @return cv::Mat
 */
cv::Mat apply_jetmap(cv::Mat image) {
    cv::Mat result = image.clone();
    if (image.channels() == 3) {
        cv::cvtColor(result, result, CV_BGR2GRAY);
    }

    result.convertTo(result, CV_8UC1);
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);

    cv::applyColorMap(result, result, cv::COLORMAP_JET);

    return result;
}

/**
 * @brief draw contour on the image
 *
 * @param img  original image
 * @param contour N*2 mat, each row contaion a 2d point in type of
 * cv::Point2d(x,y)
 * @return cloned image with drawed contour
 */
cv::Mat draw_points(cv::Mat img, cv::Mat points, cv::Scalar color) {
    assert(points.type() == CV_64FC1 && !img.empty());
    cv::Mat result = img.clone();
    if (points.empty()) {
        return result;
    }
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }

    if (result.type() != CV_8UC3) {
        result.convertTo(result, CV_8UC3);
    }

    for (int i = 0; i < points.rows; i++) {
        const cv::Vec2d& p1 = points.at<cv::Vec2d>(i);
        result.at<cv::Vec3b>(p1[1], p1[0]) =
            cv::Vec3b(color(0), color(1), color(2));
    }

    return result;
}
cv::Mat get_float_mat_vis_img(cv::Mat input) {
    cv::Mat output;
    cv::normalize(input, output, 0, 1, cv::NORM_MINMAX);
    return output;
}