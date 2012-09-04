#include "mnistfile.h"

MnistFile::MnistFile()
{
}

bool MnistFile::openFile(char *filename)
{
    file = fopen(filename, "rb");

    if(file == 0)
        return false;

    fseek(file, 0, SEEK_SET);
    if(fread(&magic, sizeof(magic), 1, file) != 1)
        return false;

    magic = swap(magic);

    if(fread(&length, sizeof(length), 1, file) != 1)
        return false;

    length = swap(length);

    switch(magic)
    {
        case 2051:
            isImages = true;

            if(fread(&rows, sizeof(rows), 1, file) != 1)
                return false;

            rows = swap(rows);

            if(fread(&cols, sizeof(cols), 1, file) != 1)
                return false;

            cols = swap(cols);

            image = new unsigned char[rows*cols];
        break;

        case 2049:
            isImages = false;

            rows = 0;
            cols = 0;
        break;

        default:
            return false;
    }

    return true;
}

bool MnistFile::closeFile()
{
    if(isImages)
    {
        delete image;
    }

    return (fclose(file) == 0);
}

bool MnistFile::readRecord(unsigned char *data)
{
    if(isImages)
    {
        if(fread(image, sizeof(unsigned char), rows*cols, file) != (unsigned) rows*cols)
            return false;

        memcpy(data, image, rows*cols*sizeof(unsigned char));
    }
    else
    {
        if(fread(&label, sizeof(unsigned char), 1, file) != 1)
            return false;

        memcpy(data, &label, sizeof(unsigned char));
    }

    return true;
}

int MnistFile::getRows()
{
    return rows;
}

int MnistFile::getCols()
{
    return cols;
}

int MnistFile::getLength()
{
    return length;
}

short MnistFile::swap(short d)
{
   short a;
   unsigned char *dst = (unsigned char *)&a;
   unsigned char *src = (unsigned char *)&d;

   dst[0] = src[1];
   dst[1] = src[0];

   return a;
}

int MnistFile::swap(int d)
{
   int a;
   unsigned char *dst = (unsigned char *)&a;
   unsigned char *src = (unsigned char *)&d;

   dst[0] = src[3];
   dst[1] = src[2];
   dst[2] = src[1];
   dst[3] = src[0];

   return a;
}

float MnistFile::swap(float d)
{
   float a;
   unsigned char *dst = (unsigned char *)&a;
   unsigned char *src = (unsigned char *)&d;

   dst[0] = src[3];
   dst[1] = src[2];
   dst[2] = src[1];
   dst[3] = src[0];

   return a;
}

double MnistFile::swap(double d)
{
   double a;
   unsigned char *dst = (unsigned char *)&a;
   unsigned char *src = (unsigned char *)&d;

   dst[0] = src[7];
   dst[1] = src[6];
   dst[2] = src[5];
   dst[3] = src[4];
   dst[4] = src[3];
   dst[5] = src[2];
   dst[6] = src[1];
   dst[7] = src[0];

   return a;
}
