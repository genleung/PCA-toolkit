#include <string>
#include <vector>
#include <memory>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

// libtclap 用来解析命令行参数
#include "tclap/CmdLine.h"

struct Parameters{
    int classFrom; ///< 类的开始id
    int classTo;    ///< 类的结束id
    int scaledWidth; ///< 缩放后的样本宽度
    int scaledHeight; ///< 缩放后的样本高度
    bool verbose;
    bool grayscaled; ///< 是否对样本图像进行灰度化
    bool contrastStretched; ///< 是否对样本进行对比度拉伸
    int binarizedType; ///< 对样本图像进行二值化, 仅对灰度图像有效; 0不二值化; 1 otsu二值化; 2 adaptive二值化
    float reservedVariance; ///< PCA保留的方差
    int edgeWidth; ///< 边缘是否移除（比如移除2个像素的周围一圈边缘）
    bool preview; ///< 是否预览样本(仅在测试时使用)
    int previewWidth = 40;
    int previewHeight = 30;
    std::string trainSamplesPath; ///< 训练样本的路径
    std::string testSamplesPath; ///< 测试样本的路径
    std::string pcaPath; ///< PCA模型的保存路径
    std::vector<int> trainSamplesCountInClasses; ///< 训练集中各个类的样本数量
    std::vector<int> testSamplesCountInClasses; ///< 测试集中各个类的样本数量
};
