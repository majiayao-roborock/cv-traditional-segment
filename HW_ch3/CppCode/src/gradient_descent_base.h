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
#include <limits>
#include <string>

class GradientDescentBase {
   public:
    /**
     * @brief Construct a new Gradient Descent Base object
     *
     * @param step_size
     */
    GradientDescentBase(double step_size);
    /**
     * @brief excutable function for the whole gradient descent method
     *
     * @param max_iteration
     */
    void run(int max_iteration);

   protected:
    /**
     * @brief initialize the parameter, which is a pure virtual function
     *
     */
    virtual void initialize() = 0;
    /**
     * @brief update the parameter, which is a pure virtual function
     *
     */
    virtual void update() = 0;
    /**
     * @brief tell if the current state fulfills the terminate condition
     *
     * @param current_iter
     * @param max_iteration
     * @return true
     * @return false
     */
    virtual bool is_terminate(int current_iter, int max_iteration) const;
    /**
     * @brief compute energy, which is a pure virtual function
     *
     * @return double
     */
    virtual double compute_energy() const = 0;
    /**
     * @brief roll back the state if energy increases, which is a pure virtual
     * function
     *
     */
    virtual void roll_back_state() = 0;
    /**
     * @brief back up state, which is a pure virtual function
     *
     */
    virtual void back_up_state() = 0;
    /**
     * @brief update the step size if energy decreases
     *
     * @param is_energy_decent
     */
    virtual void update_step_size(bool is_energy_decent);
    /**
     * @brief print terminate info, if the terminate condition fulfills
     *
     */
    virtual void print_terminate_info() const;
    /**
     * @brief return a drive class name, for example GVF, SNAKE etc.
     *
     *
     * @return std::string
     */
    virtual std::string return_drive_class_name() const = 0;

    double step_size_ = 1e-10;
    double last_energy_ = std::numeric_limits<double>::max();
};