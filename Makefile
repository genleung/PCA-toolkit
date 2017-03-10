pca: pca.cpp ../../Utils/utils.cpp ../../ML/DataProcessing.cpp
	g++ -std=c++14 pca.cpp ../../Utils/utils.cpp ../../ML/DataProcessing.cpp -o pca -lboost_filesystem -lboost_system `pkg-config --cflags --libs opencv` -I"../../Log" -L"./" -lartlog -pthread
