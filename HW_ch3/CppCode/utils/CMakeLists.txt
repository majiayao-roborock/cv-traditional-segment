
add_library(utils 
    display.cpp 
    tictoc.cpp
)
target_link_libraries(utils ${OpenCV_LIBS})