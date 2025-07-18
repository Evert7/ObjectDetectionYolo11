/**
 * @file video_inference.cpp
 * @brief Object detection in a video stream using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12. 
 * The application processes a video stream to detect objects and saves 
 * the results to a new video file with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a video stream from disk or camera.
 * - Initializing the YOLO detector with the desired model and labels.
 * - Detecting objects within each frame of the video.
 * - Drawing bounding boxes around detected objects and saving the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `videoPath`: Path to the input video file (e.g., input.mp4).
 * - `outputPath`: Path for saving the output video file (e.g., output.mp4).
 * - `modelPath`: Path to the desired YOLO model file (e.g., yolo.onnx format).
 *
 * The application can be extended to use different YOLO versions by modifying 
 * the model path and the corresponding detector class.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified video and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * @note The code includes commented-out sections to demonstrate how to switch 
 * between different YOLO models and video inputs.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */
// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "YOLO11.hpp" 
#include "Tracker.hpp"


// Thread-safe queue implementation
template <typename T>
class SafeQueue {
public:
    SafeQueue() : q(), m(), c() {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the first element from the queue.
    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = q.front();
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }


private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;

};


struct Track
{
    /* Struct to represent a Track */
    // KalmanFilter kf;
    int ID;
    int age;
    int visbleCount;
    int invisibleCount;
    std::vector<int> objectState;

};



// std::vector<int> Identity; 
// std::vector<Identity> Identities;

int IOUmin = 50;
int TLostFrames = 3;



int main()
{
    // Paths to the model, labels, input video, and output video
    const std::string labelsPath = "../models/coco.names";
    const std::string videoPath = "../data/classroom.mp4"; // Input video path
    const std::string outputPath = "../data/classroom_output.mp4"; // Output video path
    const std::string modelPath = "../models/yolo11n.onnx"; 

    // Flags for drawing warnings on the video output
    bool pedestrianDetected;
    bool vehicleDetected;
    int i = 0;

    // Initialize the YOLO detector
    bool isGPU = true; // Set to false for CPU processing
    YOLO11Detector detector(modelPath, labelsPath, isGPU); 

    // // Tracking Parameters        
    // float u_meas;
    // float v_meas;
    // float s_meas;
    // float r_meas;
    // float u_vel = 0;
    // float v_vel = 0;
    // float s__vel = 0;
    int testingCounter = 0;
    std::vector<Tracker> trackers;  // vector to store all active trackers

    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)); // Get codec of input video

    // Create a VideoWriter object to save the output video with the same codec
    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }

    // Thread-safe queues and processing...
    // Thread-safe queues
    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<std::pair<int, cv::Mat>> processedQueue;

    // Flag to indicate processing completion
    std::atomic<bool> processingDone(false);


    // Capture thread
    std::thread captureThread([&]() {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame))
        {
            frameQueue.enqueue(frame.clone()); // Clone to ensure thread safety
            frameCount++;
        }
        frameQueue.setFinished();
    });

    // Processing thread
    std::thread processingThread([&]() {
    cv::Mat frame;
    int frameIndex = 0;
    while (frameQueue.dequeue(frame))
    {
        // Detect objects in the frame
        std::vector<Detection> results = detector.detect(frame);


        vehicleDetected = false;
        pedestrianDetected = false;
        int indexToRemove = 0;

        for (const auto& det : results) {

            // Filter out all the pedestrians and vehicles
            if (det.classId == 0) {
                // std::cout << "Detected: person" << std::endl;
                pedestrianDetected = true;
            } else if (det.classId == 2 || det.classId == 3 || det.classId == 5 || det.classId == 7) {
                // std::cout << "Detected: vehicle" << std::endl;
                vehicleDetected = true;
            } else {
                if (indexToRemove >= 0 && indexToRemove < results.size()) {
                    results.erase(results.begin() + indexToRemove);
                    indexToRemove -= 1;
                }
            }
            indexToRemove += 1;


            float u_meas = det.box.x + (det.box.width / 2);         // x + w / 2.0;     horizontal pixel location of the centre of the target
            float v_meas = det.box.y + (det.box.height / 2);        // y + h / 2.0;     vertical pixel location of the centre of the target
            float s_meas = det.box.width * det.box.height;          // w * h;           scale (area of bounding box)
            float r_meas = det.box.width / float(det.box.height);   // w / h            aspect ratio 
            
            
            // (buite) Predict al die states van die trackers met die kalman filter
            // Kyk of die bb ooreenstem met n tracker deur gebruik te maak van hungarian metode
                // As hy ooreenstem
                    // Stel die age na 0 toe 
                    // Update die KF se state met die measurements 

                // As hy nie ooreenstem nie
                    // Begin n nuwe tracker ID



            if (testingCounter < 10){
                std::vector<float> detection = {u_meas, v_meas, s_meas, r_meas};
                Tracker newTracker(detection);
                int identity = newTracker.getID();
                trackers.push_back(newTracker);
                std::cout<<"Successfuly created Tracker with id: "<<identity<<std::endl;
                testingCounter++;
            }
        }

        // Loop through all the trackers and increase the age of each one, if the age is above the threshold it should be deleted
        // int TrackerToDelete = 0;
        // for (auto& tracker : trackers){
        //     if (TrackerToDelete == 0) {tracker.setAge(1);}
        //     int age = tracker.getAge();
        //     std::cout<<"Tracker with id: "<<tracker.getID()<<" has an age of "<<age<<std::endl;
        //     if (age >= TLostFrames) {
        //         trackers.erase(trackers.begin() + TrackerToDelete); 
        //     } else {
        //         tracker.increaseAge();
        //         TrackerToDelete++;
        //     }
        
        // }

        // verander die loop dat hy reg werk
        for (int i = trackers.size() - 1; i >= 0; --i) {
            if (i == 0) {trackers[i].setAge(1);}
            int age = trackers[i].getAge();
            std::cout<<"Tracker with id: "<<trackers[i].getID()<<" has an age of "<<age<<std::endl;
            if (trackers[i].getAge() >= TLostFrames) {
                std::cout << "Deleting tracker with id: " << trackers[i].getID() << std::endl;
                trackers.erase(trackers.begin() + i);
            } else {
                trackers[i].increaseAge();
            }
        }


        // Draw warnings on the frame
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.7;
        int thickness = 2;

        if (pedestrianDetected && vehicleDetected) {
            cv::putText(frame, "Warning: Pedestrian and Vehicle Detected", cv::Point(10, 60), fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
        } else if (pedestrianDetected) {
            cv::putText(frame, "Warning: Pedestrian Detected", cv::Point(10, 30), fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
        } else if (vehicleDetected) {
            cv::putText(frame, "Warning: Vehicle Detected", cv::Point(10, 60), fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
        }

        // Draw bounding boxes on the frame
        detector.drawBoundingBoxMask(frame, results); // Uncomment for mask drawing

        // Enqueue the processed frame
        processedQueue.enqueue(std::make_pair(frameIndex++, frame));
    }
    processedQueue.setFinished();
    });

    // Writing thread
    std::thread writingThread([&]() {
        std::pair<int, cv::Mat> processedFrame;
        while (processedQueue.dequeue(processedFrame))
        {
            out.write(processedFrame.second);
        }
    });

    // Wait for all threads to finish
    captureThread.join();
    processingThread.join();
    writingThread.join();

    // Release resources
    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully." << std::endl;

    return 0;
}
