# Facial Enhancement Using OpenCV and MediaPipe

This project enhances facial images captured in low quality, particularly from CCTV footage. The implementation uses MediaPipe for face detection and OpenCV for image processing techniques such as super-resolution, denoising, sharpening, and contrast enhancement.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [License](#license)

## Features

- **Face Detection**: Utilizes MediaPipe's face detection model to locate faces within images.
- **Super Resolution**: Applies the Efficient Sub-Pixel Convolutional Neural Network (ESPCN) model to enhance image resolution by 4x.
- **Image Enhancement**: Applies additional techniques such as:
  - Sharpening filters to enhance edges.
  - Denoising to reduce noise in images.
  - Histogram equalization to improve contrast.
  - Deblurring to reduce blur.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- MediaPipe
- Pre-trained ESPCN model file (`ESPCN_x4.pb`)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your_username/face-enhancement.git
   cd face-enhancement
   ```

2. Install the required packages using pip:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Download the pre-trained ESPCN model file (`ESPCN_x4.pb`) from [this link](https://github.com/Saafke/Real-ESRGAN) and place it in the project directory.

## Usage

1. Modify the `image_path` variable in the script to point to the image you want to enhance:

   ```python
   image_path = "D:\CV\3.jpeg"  # Change this path to your image file
   ```

2. Run the script:

   ```bash
   python enhance_faces.py
   ```

3. The enhanced image will be saved as `enhanced_image.jpg`, and a comparison will be displayed in a window.

## Code Overview

- **Face Detection**: The code initializes the MediaPipe face detection model, processes the input image to detect faces, and retrieves their bounding boxes.
  
- **Enhance Image Function**: The `enhance_image` function applies multiple enhancement techniques to each detected face:
  - Super-resolution is performed using the ESPCN model.
  - A sharpening filter emphasizes edges.
  - Denoising is applied to reduce color noise.
  - Histogram equalization enhances contrast.
  - A Gaussian blur is applied to reduce remaining blur.

- **Main Processing Loop**: For each detected face:
  - The face is extracted and enhanced.
  - The enhanced face is resized and placed back into the original image.
  - Bounding boxes and confidence scores are drawn on the original image.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
