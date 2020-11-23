
add_library(gdc_base gradient_descent_base.cpp)

add_library(heightmap height_map.cpp)
target_link_libraries(heightmap ${OpenCV_LIBS} lvlset)

add_library(lvlset 
    level_set_utils.cpp
    level_set_cv.cpp
    level_set_lbf.cpp

)
target_link_libraries(lvlset 
    ${OpenCV_LIBS} 
    utils 
    heightmap 
    gdc_base
)


