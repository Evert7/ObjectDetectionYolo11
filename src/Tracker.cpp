# include "Tracker.hpp"

int Tracker::NextID = 0;

Tracker::Tracker(const std::vector<float>& detection){
    id = NextID++;
}

int Tracker::getID() const {
    return id;
}