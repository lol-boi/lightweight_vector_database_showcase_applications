# Image Similarity Search Application

This application allows you to find similar images within a directory. It uses a pre-trained MobileNetV2 model to generate vector embeddings for images and a custom C++ vector database for efficient similarity searching.

## Features

- Index a folder of images to create a searchable database.
- Select a query image to find the most similar images from the indexed database.
- Simple and intuitive GUI built with Tkinter.

## Prerequisites

- **Python 3.8+**
- **Git**
- **A C++17 compatible compiler:**
  - **Windows:** Visual Studio with "Desktop development with C++" workload.
  - **macOS:** Xcode Command Line Tools.
  - **Linux:** `build-essential` or equivalent (e.g., `sudo apt-get install build-essential`).
- **CMake 3.15+:** Make sure to add CMake to your system's PATH during installation.

## Setup and Installation

### 1. Clone the Repository

First, clone the entire repository to your local machine:

```bash
git clone <repository_url>
cd showcase_custom_database
```

### 2. Build the C++ Vector Database Bindings

The core vector search functionality is a C++ library that needs to be compiled into a Python module.

```bash
# Navigate to the C++ library directory
cd lightweight_vector_database

# Create and enter a build directory
mkdir build && cd build

# Generate the build files with CMake
cmake ..

# Compile the project
# On Windows (from a developer command prompt):
cmake --build . --config Release

# On macOS and Linux:
make
```

After a successful build, the compiled Python binding (`vector_database_bindings.pyd` on Windows, `vector_database_bindings.so` on Linux/macOS) will be located in the `build` directory. The Python application is configured to find it automatically.

### 3. Set Up Python Environment

Navigate to the application directory from the root of the project:

```bash
cd ../showcase_apps/image_similarity_search_python
```

Create and activate a Python virtual environment:

```bash
# For macOS and Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Download the ONNX Model

The application uses a MobileNetV2 model for image embedding. The `mobilenetv2-7.onnx` model is expected to be in the `models/` directory within the `image_similarity_search_python` folder.

If the model is missing, you can download it from the [ONNX Model Zoo](https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx) and place it in the `models` directory.

### 5. Run the Application

Once the setup is complete, you can run the application from the `image_similarity_search_python` directory:

```bash
python main.py
```

## How to Use the Application

1.  **Index Images:**
    - Click on **"Index Images from Folder"**.
    - Select a directory containing the images you want to search through.
    - The application will process each image, generate an embedding, and add it to the database.
    - The database is automatically saved to `image_database.bin` after indexing.

2.  **Search for Similar Images:**
    - Click on **"Select Query Image"**.
    - Choose an image you want to find matches for.
    - The application will display the query image on the left and the most similar images from the database on the right.
    - You can change the number of results (`k`) to retrieve.

3.  **Database Management:**
    - **Load Database:** Loads an existing `image_database.bin` file.
    - **Save Database:** Manually saves the current state of the image index.