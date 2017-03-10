# PCA-toolkit
A toolkit for pre-process image-samples into LibSVM training &amp; testing data format

## Introduction
This toolkit requires opencv3.x and libtclap-dev(for parsing command-line arguments). All the images to be pre-processed are tiny pictures about 32x32 or 24x24 (or other sizes), and placed in the './data/' directory. The './data' directory hierachy is as follows:

    data_empy/
    ├── test
    │   ├── 1 -> digits/0
    │   ├── 10 -> digits/9
    │   ├── 11 -> symbols/add
    │   ├── 12 -> symbols/sub
    │   ├── 13 -> symbols/mul
    │   ├── 14 -> symbols/div
    │   ├── 15 -> symbols/equal
    │   ├── 16 -> symbols/question
    │   ├── 17 -> symbols/lbracket
    │   ├── 18 -> symbols/rbracket
    │   ├── 2 -> digits/1
    │   ├── 3 -> digits/2
    │   ├── 4 -> digits/3
    │   ├── 5 -> digits/4
    │   ├── 6 -> digits/5
    │   ├── 7 -> digits/6
    │   ├── 8 -> digits/7
    │   ├── 9 -> digits/8
    │   ├── digits
    │   │   ├── 0
    │   │   ├── 1
    │   │   ├── 2
    │   │   ├── 3
    │   │   ├── 4
    │   │   ├── 5
    │   │   ├── 6
    │   │   ├── 7
    │   │   ├── 8
    │   │   └── 9
    │   ├── README
    │   └── symbols
    │       ├── add
    │       ├── div
    │       ├── equal
    │       ├── lbracket
    │       ├── mul
    │       ├── question
    │       ├── rbracket
    │       └── sub
    └── train
        ├── 1 -> digits/0
        ├── 10 -> digits/9
        ├── 11 -> symbols/add
        ├── 12 -> symbols/sub
        ├── 13 -> symbols/mul
        ├── 14 -> symbols/div
        ├── 15 -> symbols/equal
        ├── 16 -> symbols/question
        ├── 17 -> symbols/lbracket
        ├── 18 -> symbols/rbracket
        ├── 2 -> digits/1
        ├── 3 -> digits/2
        ├── 4 -> digits/3
        ├── 5 -> digits/4
        ├── 6 -> digits/5
        ├── 7 -> digits/6
        ├── 8 -> digits/7
        ├── 9 -> digits/8
        ├── digits
        │   ├── 0
        │   ├── 1
        │   ├── 2
        │   ├── 3
        │   ├── 4
        │   ├── 5
        │   ├── 6
        │   ├── 7
        │   ├── 8
        │   └── 9
        ├── README
        └── symbols
            ├── add
            ├── div
            ├── equal
            ├── lbracket
            ├── mul
            ├── question
            ├── rbracket
            └── sub

Notice that only './data/train/[label-numbers]' & './data/test/[label-numbers]' are matter。Those 'label-numbers' (in the previous case is ranging 0~18) are class-label-numbers when doing classification with LibSVM。You can set the lower-upper bounds with this toolkit (parameter -f 1 -t 18 will do)。

## Compiling & Run
This toolkit only tested under ubuntu16.04.   

    $ make

And then, you can run the toolkit now.
    
    $ ./pca -f 1 -t 18 -e2 -v 0.99

This would generate a [pca].xml file under './data/', a training [train].dat file under './data/training/' and a testing [test].dat file under './data/test/'. You can put the train.dat & test.data into LibSVM's 'tools' directory, and run with easy.py:

    ./easy.py train.dat test.dat

easy.py will do a massive cross-validation and test procedure to find the best parameters for SVM classfication.
