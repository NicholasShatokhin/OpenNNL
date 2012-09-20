#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <ctime>

inline void initialize_random_generator()
{
    srand(time(NULL));
}

inline double closed_interval_rand(double x0, double x1)
{
    return x0 + (x1 - x0) * rand() / ((double) RAND_MAX);
}

inline double unified_random()
{
    return closed_interval_rand(0, 2) - 1;
}

#endif // UTILS_H
