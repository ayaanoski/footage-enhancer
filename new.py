import cv2
import mediapipe as mp
import numpy as np

# Initialize face detection with MediaPipe
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

# Initialize Super Resolution Model (ESPCN)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x4.pb")  # You will need to download pre-trained model files
sr.setModel("espcn", 4)  # Using ESPCN model with 4x upscaling


def enhance_image(image):
    """Enhance image using super-resolution, denoising, and other methods"""

    # Apply 4x Super-Resolution
    result = sr.upsample(image)

    # Apply sharpening filter (emphasize edges to make features stand out)
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel_sharpen)

    # Noise reduction
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)

    # Apply histogram equalization to improve contrast
    img_yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Deblur (using Gaussian filter here, can be modified for more advanced methods)
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


# Load the image
image_path = "D:\\CV\\3.jpeg"  # Change this path to your image file
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
else:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            face = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

            # Enhance face image
            enhanced_face = enhance_image(face)

            # Resize enhanced face to match the original bounding box size
            enhanced_face_resized = cv2.resize(enhanced_face, (bbox[2], bbox[3]))

            # Place the resized enhanced image back into the original image
            img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = (
                enhanced_face_resized
            )

            # Draw the bounding box and score on the image
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(detection.score[0] * 100)}%",
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2,
            )

    # Save the enhanced image comparison for future evaluation
    cv2.imwrite("enhanced_image.jpg", img)

    # Display the original and enhanced image side by side
    img_comparison = np.hstack((cv2.imread(image_path), img))
    cv2.imshow("Comparison", img_comparison)

    # Wait for user to press a key, then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
