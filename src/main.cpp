
// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "YOLO11.hpp" // Ensure YOLO11.hpp or other version is in your include path

int main()
{
    // Configuration parameters
    const std::string labelsPath = "../models/coco.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n.onnx";     // Path to YOLO11 model
    const std::string imagePath  = "../test.jpg";           // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 detector
    YOLO11Detector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object detection to get bboxs
    std::vector<Detection> detections = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBoxMask(image, detections);

    // Display the annotated image
    cv::imshow("YOLO11 Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}

