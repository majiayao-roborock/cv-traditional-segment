/**
______________________________________________________________________
*********************************************************************
* @brief This file is developed for the course of ShenLan XueYuan:
* Fundamental implementations of Computer Vision
* all rights preserved
* @author Xin Jin, Zhaoran Wu
* @contact: xinjin1109@gmail.com, zhaoran.wu1@gmail.com
* @discription: This file is used for some common utils by CV and LBF Model
*
______________________________________________________________________
*********************************************************************
**/
#pragma once
#include "height_map.h"
#include "level_set_cv.h"
#include <opencv2/core.hpp>
/**
 * @brief Do sobel filter to detect horizontal or vertical edges
 *
 * @param im
 * @param flag flag = 0, do sobel x, flag = 1, do sobel y
 * @return cv::Mat
 */
cv::Mat do_sobel(cv::Mat im, int flag);
/**
 * @brief Create the gaussian kernel
 *
 * @param size
 * @param sigma
 * @return cv::Mat
 */
cv::Mat get_gaussian_kernel(int size, double sigma);

/**
 * @brief Compute laplacian of height_map.get_map()
 *
 * @param height_map
 * @return cv::Mat
 */
cv::Mat compute_laplacian_map(const HeightMap& height_map);

/**
 * @brief Get the sub image w.r.t the position (row,col)
 *
 * @param image
 * @param row
 * @param col
 * @param window_size
 * @return cv::Mat
 */
cv::Mat get_sub_image(cv::Mat image, int row, int col, int window_size);

/**
 * @brief Compute |grad(Matrix)| for the use of regularization energy term
 *
 * @param mat
 * @return cv::Mat
 */
cv::Mat compute_mat_grad_magnitude(cv::Mat mat);

/**
 * @brief  Heaviside function H(z) = 0.5(1+2/pi*arctan(z/eps)) for each element
 * of height_map.get_map()
 *
 * @param height_map
 * @param eps epsilon controls the stepness of the function
 * @return cv::Mat
 */

cv::Mat heaviside(const HeightMap& height_map, double eps = 1.0);
/**
 * @brief Compute the square difference of two matrix
 * (img1-img2)^2 for each element
 * @param img1
 * @param img2
 * @return cv::Mat
 */
cv::Mat compute_square_diff(cv::Mat img1, cv::Mat img2);

/**
 * @brief 1-H(z)
 *
 * @param height_map
 * @param eps
 * @return cv::Mat
 */
cv::Mat complementary_heaviside(const HeightMap& height_map, double eps = 1.0);

/**
 * @brief Gradient of Heaviside function delta(z) = eps/(pi*(eps^2 + z^2))
 *
 * @param height_map
 * @param eps
 * @return cv::Mat
 */
cv::Mat dirac(const HeightMap& height_map, double eps = 1.0);

/**
 * @brief div(grad(phi)/|grad(phi|)
 *
 * @param height_map : phi
 * @return cv::Mat
 */
cv::Mat compute_div_delta_map(const HeightMap& height_map);

/**
 * @brief Update of image force (data term),which is
 * dirac(phi)*(lamda1*(I-cf)^2-lamda2*(I-cb)^2)
 *
 * @param height_map : get phi, which is from height_map.get_map()
 * @param original_image : I
 * @param weight_foreground : lamda1
 * @param weight_background : lamda2
 * @param center_foreground : cf
 * @param center_background : cb
 * @param eps
 * @return cv::Mat
 */
cv::Mat compute_derivative_data_term(const HeightMap& height_map,
                                     cv::Mat original_image,
                                     double weight_foreground,
                                     double weight_background,
                                     double center_foreground,
                                     double center_background, double eps);

/**
 * @brief Update the length term mu*dirac(phi)*div(grad(phi)/|grad(phi|)
 *
 * @param height_map
 * @param eps
 * @return cv::Mat
 */
cv::Mat compute_derivative_length_term(const HeightMap& height_map, double eps);

/**
 * @brief Update gradient term, which keep the gradient of height map one
 * miu*(laplace(phi) - div(grad(phi)/|grad(phi|))
 * @param height_map
 * @return cv::Mat
 */
cv::Mat compute_derivative_gradient_term(const HeightMap& height_map);

/**
 * @brief Update the center see slide page 21
 *
 * @return cv::Mat
 */
double compute_center(cv::Mat img, const HeightMap& height_map, double eps,
                      bool is_background);

/**
 * @brief Update center in window for LBF Model
 *
 * @param row
 * @param col
 * @param size
 * @param gauss_kernel
 * @param img
 * @param height_map
 * @param eps
 * @param is_background
 * @return double
 */
double compute_center_in_window(int row, int col, int size,
                                cv::Mat gauss_kernel, cv::Mat img,
                                const HeightMap& height_map, double eps,
                                bool is_background);

/**
 * @brief compute data term energy see slide page 17
 *
 * @param height_map
 * @param original_image
 * @param weight_foreground
 * @param weight_background
 * @param center_foreground
 * @param center_background
 * @param eps
 * @return double
 */
double compute_data_term_energy(const HeightMap& height_map,
                                cv::Mat original_image,
                                double weight_foreground,
                                double weight_background,
                                double center_foreground,
                                double center_background, double eps);

/**
 * @brief : compute length term energy see slide page 17
 *
 * @param height_map
 * @param eps
 * @return double
 */
double compute_length_term_energy(const HeightMap& height_map, double eps);

/**
 * @brief compute gradient preserve energy see slide page 17
 *
 * @param height_map
 * @return double
 */
double compute_gradient_preserve_energy(const HeightMap& height_map);

/**
 * @brief visualize the level set segementation result
 *
 * @param origin_img : origin image in the type of CV_8UC3
 * @param phi: level set function phi
 * @param delay: opencv waitkey param, set 0 if want to stop at each iteration
 */
void visualize_lvl_set_segemenation(cv::Mat origin_img, const HeightMap& phi,
                                    int delay = 0);
/**
 * @brief visualize the update step of the level set
 *
 * @param update_step_data_term
 * @param update_step_length_term
 * @param update_step_gradient_term
 * @param delay:opencv waitkey param, set 0 if want to stop at each iteratio
 */
void visualize_lvl_set_update_term(cv::Mat update_step_data_term,
                                   cv::Mat update_step_length_term,
                                   cv::Mat update_step_gradient_term,
                                   int delay);