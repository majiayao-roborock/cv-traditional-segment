add_executable(test_cv test_cv.cpp)
target_link_libraries(test_cv 
    ${OpenCV_LIBS}
    lvlset
    heightmap
    utils
)


add_executable(test_lbf test_lbf.cpp)
target_link_libraries(test_lbf 
    ${OpenCV_LIBS}
    lvlset
    heightmap
    utils
)

add_executable(test_height_map test_height_map.cpp)
target_link_libraries(test_height_map 
    ${OpenCV_LIBS}
    lvlset
    heightmap
    utils
)


