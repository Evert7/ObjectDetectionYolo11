# include "Tracker.hpp"
#include <iostream>

int Tracker::NextID = 0;

Tracker::Tracker(const std::vector<float>& detection){
    id = NextID++;
    initializeKF(detection);
}

int Tracker::getID() const {
    return id;
}

int Tracker::getAge() const{
    return age;
}

void Tracker::increaseAge() {
    age++;
}

void Tracker::setAge(int newAge) {
    age = newAge;
}

void Tracker::predict() {
    state = F *state;
    P = F * P * F.transpose() + Q;
    // age++;
}

void Tracker::initializeKF(const std::vector<float>& detection) {
    state = Eigen::VectorXf::Zero(StateDim);
    state(0) = detection[0];  // u
    state(1) = detection[1];  // v
    state(2) = detection[2];  // s
    state(3) = detection[3];  // r

    P = Eigen::MatrixXf::Identity(StateDim, StateDim);

    // F matrix
    F = Eigen::MatrixXf::Identity(StateDim, StateDim);
    F(0,4) = 1.0f; // u_dot
    F(1,5) = 1.0f; // u_dot
    F(2,6) = 1.0f; // u_dot

    // H matrix
    H = Eigen::MatrixXf::Zero(measDim, StateDim);
    H(0, 0) = 1.0f;  // u
    H(1, 1) = 1.0f;  // v
    H(2, 2) = 1.0f;  // s
    H(3, 3) = 1.0f;  // r

    // Q matrix
    Q = Eigen::MatrixXf::Identity(StateDim, StateDim) * 1e-2;

    // R matrix 
    R = Eigen::MatrixXf::Identity(measDim, measDim) * 1e-1;
}

void Tracker::update(const std::vector<float>& detection) {
    Eigen::VectorXf z(measDim);
    z << detection[0], detection[1], detection[2], detection[3];

    Eigen::VectorXf y = z - H * state;
    Eigen::MatrixXf S = H * P * H.transpose() + R;
    Eigen::MatrixXf K = P * H.transpose() * S.inverse();

    // Update state
    state = state + K * y;

    // Update covariance
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(StateDim, StateDim);
    P = (I - K * H) * P;
}

std::vector<float> Tracker::getState() const {
    return { state(0), state(1), state(2), state(3) };
}
