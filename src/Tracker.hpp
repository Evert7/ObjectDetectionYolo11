#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>


class Tracker{
    public:
        // Code to create a new Tracker 
        Tracker(const std::vector<float>& detection);
        int getID() const;
        int getAge() const;
        void increaseAge();
        void setAge(int newAge);

        // Kalman filter code
        void predict();
        void update(const std::vector<float>& detection);
        std::vector<float> getState() const;

    private:
        // Tracker code
        static int NextID;
        int id;
        int age = 0;

        // Kalman filer code 
        Eigen::VectorXf state;
        Eigen::MatrixXf P;

        static constexpr int StateDim = 7;
        static constexpr int measDim = 4;

        Eigen::MatrixXf F;  
        Eigen::MatrixXf H;  
        Eigen::MatrixXf Q;  
        Eigen::MatrixXf R; 

        void initializeKF(const std::vector<float>& detection);

};

#endif