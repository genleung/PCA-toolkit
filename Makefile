pca: pca.cpp 
	g++ -std=c++14 pca.cpp -o pca -lboost_filesystem -lboost_system `pkg-config --cflags --libs opencv` -lcflog -pthread
