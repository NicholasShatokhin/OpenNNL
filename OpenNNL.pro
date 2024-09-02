TEMPLATE = app
CONFIG += console
CONFIG -= qt

LIBS += -lrt

SOURCES += src/main.cpp \
    src/opennnl.cpp \
    src/mnistfile.cpp

HEADERS += \
    src/opennnl.h \
    src/utils.h \
    src/mnistfile.h

OTHER_FILES += \
    data/mnist/train-labels.idx1-ubyte \
    data/mnist/train-images.idx3-ubyte \
    data/mnist/t10k-labels.idx1-ubyte \
    data/mnist/t10k-images.idx3-ubyte

