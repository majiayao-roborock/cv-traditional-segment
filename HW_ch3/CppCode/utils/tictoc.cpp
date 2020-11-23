#include "tictoc.h"

namespace tictoc {

Tictoc& Tictoc::getInstance() {
    static Tictoc instance;
    return instance;
}

void Tictoc::tic() {
    tLast_ = std::chrono::high_resolution_clock::now();
}

int Tictoc::toc() {
    tCurr_ = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(tCurr_ - tLast_).count();
}

void Tictoc::tic(const std::string& name) {
    tLastMap_[name] = std::chrono::high_resolution_clock::now();
}

int Tictoc::toc(const std::string& name) {
    tCurr_ = std::chrono::high_resolution_clock::now();
    if (tLastMap_.count(name) > 0) {
        return std::chrono::duration_cast<std::chrono::microseconds>(tCurr_ - tLastMap_[name]).count();
    } else {
        return -1;
    }
}

void tic() {
    Tictoc::getInstance().tic();
}

int toc() {
    return Tictoc::getInstance().toc();
}

void tic(const std::string& name) {
    Tictoc::getInstance().tic(name);
}

int toc(const std::string& name) {
    return Tictoc::getInstance().toc(name);
}

} // namespace tictoc
