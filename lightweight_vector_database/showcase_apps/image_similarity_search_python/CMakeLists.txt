# CMakeLists.txt for the Python Image Similarity Search application

# This project doesn't build C++ executables directly, but depends on the
# vector_database_bindings Python module.

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create a dummy target for the Python application to manage dependencies
add_custom_target(image_similarity_search_python_app ALL
    DEPENDS vector_database_bindings
)

# Install the Python module alongside the Python application
install(TARGETS vector_database_bindings
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages
        RENAME vector_database_bindings${PYTHON_EXTENSION}
)

# Install the Python application files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/showcase_apps/image_similarity_search_python
        FILES_MATCHING PATTERN "*.py"
        PATTERN "*.txt"
)