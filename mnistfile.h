#ifndef MNISTFILE_H
#define MNISTFILE_H

#include <cstdio>
#include <cstring>

class MnistFile
{
public:
    MnistFile();

    bool openFile(char * filename);
    bool closeFile();

    bool readRecord(unsigned char * data);

    int getRows();
    int getCols();
    int getLength();

protected:
    FILE * file;

    int magic;
    int length;
    int rows;
    int cols;
    bool isImages;

    short swap(short d);
    int swap(int d);
    float swap(float d);
    double swap(double d);

    unsigned char * image;
    unsigned char label;
};

#endif // MNISTFILE_H
