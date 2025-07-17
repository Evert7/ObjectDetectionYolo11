#ifndef TRACKER_H
#define TRACKER_H

#include <vector>

class Tracker{
    public:
        Tracker(const std::vector<float>& detection);
        int getID() const;



    private:
        static int NextID;
        int id;

};

#endif