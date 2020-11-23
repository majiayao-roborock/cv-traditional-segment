#pragma once

#include <chrono>
#include <map>
#include <string>

namespace tictoc {
/**
 * \brief Singleton class that provides tic toc functionality from Matlab to stop time
 */
class Tictoc {
public:
    // singleton
    static Tictoc& getInstance();
    Tictoc(Tictoc const&) = delete;
    void operator=(Tictoc const&) = delete;

    /**
     *  \brief sets timer to current time
     */
    void tic();

    void tic(const std::string& name);

    /**
     *  \brief stop time
     *  \return time in microseconds since tic() was called
     */
    int toc();

    int toc(const std::string& name);

private:
    Tictoc(){};

    std::chrono::high_resolution_clock::time_point tLast_, tCurr_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> tLastMap_;
};

void tic();
int toc();
void tic(const std::string& name);
int toc(const std::string& name);

} // namespace tictoc
