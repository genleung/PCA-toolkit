# PCA-toolkit
A toolkit for pre-processing image-samples into LibSVM training &amp; testing data format

## Requirements
This toolkit only tested under ubuntu16.04. It requires opencv3.x (opencv2.x maybe works, but not tested), libtclap-dev (for parsing command-line arguments) and libartLog (https://github.com/genleung/artLog ). You can install them as follows.

    $ sudo apt install libopencv-dev libtclap-dev  # if opencv2.x doesnt work, do install opencv3.x manually.
    $ cd ~
    $ git clone https://github.com/genleung/artLog 
    $ cd artLog && mkdir build && cd build && cmake ..
    
And then copy files into filesystem. 
    
    $ cd ~/artLog/src
    $ sudo cp Log.h LogStream.h /usr/local/include
    $ cd ~/artLog/build/lib
    $ sudo cp * /usr/local/lib/
    $ rm -rf ~/artLog  # remove artLog, if you dont need it anymore
    
## Compiling & Run

After that, you can now proceed to install PCA-toolkit.

    $ cd ~
    $ git clone https://github.com/genleung/PCA-toolkit
    $ cd PCA-toolkit
    $ make 

If no errors emerges, you can run the toolkit now.
    
    $ ./pca -f 1 -t 18 -e2 -v 0.99

This would generate a [pca].xml file under './data/', a training [train].dat file under './data/training/' and a testing [test].dat file under './data/test/'. You can put the train.dat & test.data into LibSVM's 'tools' directory, and run with easy.py:

    ./easy.py train.dat test.dat

easy.py will do a massive cross-validation and test procedure to find the best parameters for SVM classfication.    

## Details
All the images to be pre-processed are tiny pictures about 32x32 or 24x24 (or other sizes), and placed in the './data/' directory. The './data' directory hierachy is as follows:

    data/
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

Notice that only './data/train/[label-numbers]' & './data/test/[label-numbers]' are important。Those 'label-numbers' ( ranging 1~18 in the previous case) are class-labels when doing classification with LibSVM. You can set the lower&upper bounds with this toolkit (parameter -f 1 -t 18 will do)。


