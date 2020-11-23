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
#include "level_set_utils.h"
#include "display.h"
#include <opencv2/imgproc.hpp>

cv::Mat do_sobel(cv::Mat input, int flag = 0) {
    cv::Mat im = input.clone();
    if (im.channels() != 1) {
        cv::cvtColor(input, im, CV_BGR2GRAY);
    }
    if (im.type() != CV_64FC1) {
        im.convertTo(im, CV_64FC1);
    }

    cv::Mat output(im.size(), im.type());
    for (int r = 0; r < im.rows; r++) {
        for (int c = 0; c < im.cols; c++) {
            int r_dhs = r + flag;
            int c_rhs = c + (1 - flag);
            c_rhs = std::min(std::max(0, c_rhs), im.cols - 1);
            r_dhs = std::min(std::max(0, r_dhs), im.rows - 1);
            int r_uhs = r - flag;
            int c_lhs = c - (1 - flag);
            c_lhs = std::min(std::max(0, c_lhs), im.cols - 1);
            r_uhs = std::min(std::max(0, r_uhs), im.rows - 1);
            output.at<double>(r, c) = 0.5 * (im.at<double>(r_dhs, c_rhs) -
                                             im.at<double>(r_uhs, c_lhs));
        }
    }
    return output;
}

cv::Mat get_gaussian_kernel(int size, double sigma) {
    assert(size % 2 == 1);
    cv::Point center((size - 1) / 2, (size - 1) / 2);
    cv::Mat result = cv::Mat::zeros(cv::Size(size, size), CV_64F);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            result.at<double>(r, c) =
                (M_1_PI * 0.5 / (sigma * sigma)) *
                exp(-(pow(r - center.x, 2) + pow(c - center.y, 2)) /
                    (2 * sigma * sigma));
        }
    }
    return result / cv::sum(result)[0];
}

cv::Mat compute_laplacian_map(const HeightMap& sdf_map) {
    cv::Mat result;
    cv::Laplacian(sdf_map.get_map(), result, CV_64F, 3, cv::BORDER_REPLICATE);
    return result;
}

cv::Mat compute_square_diff(cv::Mat img1, cv::Mat img2) {
    cv::Mat diff = img1 - img2;
    cv::Mat square_diff;
    cv::pow(diff, 2.0f, square_diff);
    return square_diff;
}

cv::Mat compute_div_delta_map(const HeightMap& sdf_map) {
    cv::Mat phi = sdf_map.get_map();
    cv::Mat phi_grad_mag = compute_mat_grad_magnitude(phi);
    cv::Mat d_phi_dx = do_sobel(phi, 0);
    cv::Mat d_phi_dy = do_sobel(phi, 1);

    d_phi_dx.mul(1.0 /
                 (phi_grad_mag + 1e-8 * cv::Mat::ones(phi.size(), phi.type())));
    d_phi_dy.mul(1.0 /
                 (phi_grad_mag + 1e-8 * cv::Mat::ones(phi.size(), phi.type())));

    d_phi_dx = do_sobel(d_phi_dx, 0);
    d_phi_dy = do_sobel(d_phi_dy, 1);

    return (d_phi_dx + d_phi_dy);
}

inline double heaviside(double z, double eps) {
    return 0.5 * (1 + M_2_PI * atan2(z, eps));
}

inline double dirac(double z, double eps = 1.0) {
    return eps * M_1_PI / (std::pow(eps, 2.0) + std::pow(z, 2.0));
}

cv::Mat heaviside(cv::Mat map, double eps) {
    cv::Mat result;
    // todo implement heaviside function w.r.t phi
    return result;
}

cv::Mat heaviside(const HeightMap& sdf_map, double eps) {
    return heaviside(sdf_map.get_map(), eps);
}

cv::Mat complementary_heaviside(const HeightMap& sdf_map, double eps) {
    cv::Mat h_phi = heaviside(sdf_map, eps);
    return cv::Mat::ones(h_phi.size(), h_phi.type()) - h_phi;
}

cv::Mat complementary_heaviside(cv::Mat mat, double eps) {
    cv::Mat h_phi = heaviside(mat, eps);
    return cv::Mat::ones(h_phi.size(), h_phi.type()) - h_phi;
}

cv::Mat dirac(const HeightMap& sdf_map, double eps) {
    cv::Mat result;
    // todo implement dirac function w.r.t. phi
    return result;
}

cv::Mat compute_derivative_data_term(const HeightMap& sdf_map,
                                     cv::Mat original_image,
                                     double weight_foreground,
                                     double weight_background,
                                     double center_foreground,
                                     double center_background, double eps) {
    cv::Mat result, e_foreground, e_background;
    // todo implement update of data term see slide page 21

    cv::Mat vis;
    cv::Mat vis_dirac = get_float_mat_vis_img(dirac(sdf_map, eps));
    cv::Mat vis_e_foreground = get_float_mat_vis_img(e_foreground);
    cv::Mat vis_e_background = get_float_mat_vis_img(e_background);

    cv::hconcat(vis_dirac, vis_e_foreground, vis);
    cv::hconcat(vis, vis_e_background, vis);
    disp_image(vis, "top: dirac, mid : e_foregroud, down: e_backgroud", 1);

    return result;
}

cv::Mat compute_derivative_length_term(const HeightMap& sdf_map, double eps) {
    cv::Mat result;
    // todo implement update of length term see slide page 21
    return result;
}

cv::Mat compute_derivative_gradient_term(const HeightMap& sdf_map) {
    cv::Mat result;
    // todo implement update of gradient preserve term, the gradient of sdf
    // todo should be closed to one
    return result;
}

double compute_center(cv::Mat img, const HeightMap& height_map, double eps,
                      bool is_background) {
    if (is_background) {
        return img.dot(heaviside(height_map, eps)) /
               cv::sum(heaviside(height_map, eps))[0];
    } else {
        return img.dot(complementary_heaviside(height_map, eps)) /
               cv::sum(complementary_heaviside(height_map, eps))[0];
    }
}

double compute_center_in_window(int row, int col, int size,
                                cv::Mat gauss_kernel, cv::Mat img,
                                const HeightMap& height_map, double eps,
                                bool is_background) {
    cv::Mat roi = get_sub_image(img, row, col, size);
    // cv::Mat vis_roi = get_float_mat_vis_img(roi);
    // disp_image(vis_roi, "vis", 0);
    cv::Mat roi_height_map =
        get_sub_image(height_map.get_map(), row, col, size);
    if (is_background) {
        return roi.dot(heaviside(roi_height_map, eps)) /
               cv::sum(heaviside(roi_height_map, eps))[0];
    } else {
        return roi.dot(complementary_heaviside(roi_height_map, eps)) /
               cv::sum(complementary_heaviside(roi_height_map, eps))[0];
    }
}
cv::Mat compute_mat_grad_magnitude(cv::Mat mat) {
    cv::Mat mat_dev_x = do_sobel(mat, 0);
    cv::Mat mat_dev_y = do_sobel(mat, 1);
    cv::Mat mat_grad_square =
        (mat_dev_x.mul(mat_dev_x) + mat_dev_y.mul(mat_dev_y));
    cv::Mat mat_grad_magnitude;
    cv::sqrt(mat_grad_square, mat_grad_magnitude);
    return mat_grad_magnitude;
}

double compute_data_term_energy(const HeightMap& height_map,
                                cv::Mat original_image,
                                double weight_foreground,
                                double weight_background,
                                double center_foreground,
                                double center_background, double eps) {
    double result;
    // todo implement data term energy see slide page 19
    return result;
}

double compute_length_term_energy(const HeightMap& height_map, double eps) {
    double result;
    // todo implement length term energy see slide page 19
    return result;
}

double compute_gradient_preserve_energy(const HeightMap& height_map) {
    double result;
    // todo implement gradient preserve energy see slide page 19
    return result;
}
cv::Mat get_sub_image(cv::Mat image, int row, int col, int window_size) {
    cv::Rect img_rect = cv::Rect(cv::Point(0, 0), image.size());
    cv::Rect roi =
        cv::Rect(cv::Point(col - window_size / 2, row - window_size / 2),
                 cv::Size(window_size, window_size));
    cv::Rect intersection = img_rect & roi;

    cv::Mat sub_img = cv::Mat::zeros(intersection.size(), image.type());
    image(intersection).copyTo(sub_img);
    return sub_img;
}

void visualize_lvl_set_segemenation(cv::Mat origin_img, const HeightMap& phi,
                                    int delay) {
    cv::Mat vis_sdf_draw = draw_height_map(phi);
    cv::Mat vis_sdf_with_contour = draw_points(
        vis_sdf_draw, phi.get_contour_points(), cv::Scalar(255, 255, 255));

    cv::Mat vis_seg_image = origin_img.clone();
    vis_seg_image = draw_points(vis_seg_image, phi.get_contour_points(),
                                cv::Scalar(0, 0, 255));

    cv::Mat vis_label = phi.get_fore_background_label_map();
    vis_label.convertTo(vis_label, CV_8UC1);
    cv::cvtColor(vis_label, vis_label, CV_GRAY2BGR);

    cv::Mat vis;
    cv::hconcat(vis_sdf_with_contour, vis_seg_image, vis);
    cv::hconcat(vis, vis_label, vis);

    cv::imshow("left: phi set, mid: seg on original image, right : seg label ",
               vis);
    cv::waitKey(delay);
}

void visualize_lvl_set_update_term(cv::Mat update_step_data_term,
                                   cv::Mat update_step_length_term,
                                   cv::Mat update_step_gradient_term,
                                   int delay) {
    cv::Mat vis;
    cv::Mat vis_update_data_term = get_float_mat_vis_img(update_step_data_term);
    cv::Mat vis_update_lenght_term =
        get_float_mat_vis_img(update_step_length_term);
    cv::Mat vis_update_graient_term =
        get_float_mat_vis_img(update_step_gradient_term);

    cv::hconcat(update_step_data_term, update_step_length_term, vis);
    cv::hconcat(vis, update_step_gradient_term, vis);

    cv::imshow(
        "left : data term update , mid : lenght term update , right : gradient "
        "term update",
        vis);
    cv::waitKey(delay);
}