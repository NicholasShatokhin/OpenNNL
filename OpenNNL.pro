TEMPLATE = app
CONFIG += console
CONFIG -= qt

LIBS += -lrt

SOURCES += main.cpp \
    opennnl.cpp \
    mnistfile.cpp

HEADERS += \
    opennnl.h \
    utils.h \
    mnistfile.h

OTHER_FILES += \
    data/mnist/train-labels.idx1-ubyte \
    data/mnist/train-images.idx3-ubyte \
    data/mnist/t10k-labels.idx1-ubyte \
    data/mnist/t10k-images.idx3-ubyte

