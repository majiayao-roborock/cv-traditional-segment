#pragma once
#include <opencv2/core.hpp>

class HeightMap {  // phi
   public:
    /**
     * @brief Construct a new HeightMap object : sign distance function with a
     * circular zero level set
     *
     * @param rows
     * @param cols
     * @param center
     * @param radius
     */
    HeightMap(int rows, int cols, cv::Point center, double radius);
    /**
     * @brief Construct a new HaightMap object : sign distance function with a
     * rectangular zero level set
     *
     * @param rows
     * @param cols
     */
    HeightMap(int rows, int cols);
    /**
     * @brief Get the map object, which is a private member variable
     *
     * @return cv::Mat
     */
    cv::Mat get_map() const;
    /**
     * @brief : update map_
     *
     * @param step
     */
    void add(cv::Mat step);
    /**
     * @brief Return a N*2 mat, each row is a point2d(x,y);
     *
     * @return cv::Mat
     */
    cv::Mat get_contour_points() const;
    /**
     * @brief Get the gradient magnitude level set object, which is get
     * |grad(phi)|
     *
     * @return double
     */
    double get_gradient_magnitude_level_set();
    /**
     * @brief Get the fore background label map object, which gives a
     * segmentation result
     *
     * @return cv::Mat
     */
    cv::Mat get_fore_background_label_map() const;

   private:
    cv::Mat map_;  // cv_64F
};