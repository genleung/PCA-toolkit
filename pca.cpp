#include "pca.h"
#include "Log.h"

namespace fs = boost::filesystem;


// The default variance to maintain
const float DEFAULT_RESERVED_VARIANCE = 0.97;

const std::string DEFAULT_TRAIN_SAMPLE_PATH="data/train";
const std::string DEFAULT_TEST_SAMPLE_PATH="data/test";
const std::string DEFAULT_PCA_PATH="data";

// 参数集合
Parameters g_param;

bool parseCommandLine(int argc, char* argv[]){
    Parameters* param=&g_param;
    std::shared_ptr<TCLAP::ValueArg<int>> classFrom, classTo; // 类label区间表达：[classFrom, classTo] (闭区间)
    std::shared_ptr<TCLAP::ValueArg<int>> width, height; // 对样本进行缩放，此处指定缩放后的尺寸；若为非正值，则不缩放
    std::shared_ptr<TCLAP::ValueArg<int>> grayscaled; // 灰度化
    std::shared_ptr<TCLAP::ValueArg<int>> contrastStretched; // 对比度拉伸
    std::shared_ptr<TCLAP::ValueArg<int>> binarizedType; // 对样本进行二值化：0不二值化; 1 otsu二值化; 2 adaptive二值化
    std::shared_ptr<TCLAP::ValueArg<float>> reservedVariance;
    std::shared_ptr<TCLAP::ValueArg<std::string>> trainSamplesPath;
    std::shared_ptr<TCLAP::ValueArg<std::string>> testSamplesPath;
    std::shared_ptr<TCLAP::ValueArg<std::string>> pcaPath;
    std::shared_ptr<TCLAP::ValueArg<int>> edgeWidth;
    std::shared_ptr<TCLAP::SwitchArg> preview;
    std::shared_ptr<TCLAP::SwitchArg> verbose;

    if(param==nullptr) return false;

    try{
        TCLAP::CmdLine cmd("PCA data dimension reduction toolkit from ChangFeng Info Tech", ' ', "0.1");

        trainSamplesPath=std::make_shared<TCLAP::ValueArg<std::string>>("R", "train", "Path of train-samples", false, DEFAULT_TRAIN_SAMPLE_PATH, "train_samples_path", cmd);
        testSamplesPath=std::make_shared<TCLAP::ValueArg<std::string>>("T", "test", "Path of test-samples", false, DEFAULT_TEST_SAMPLE_PATH, "test_samples_path", cmd);
        pcaPath=std::make_shared<TCLAP::ValueArg<std::string>>("P", "pca", "Path of PCA model", false, DEFAULT_PCA_PATH, "PCA_file_path", cmd);

        preview=std::make_shared<TCLAP::SwitchArg>("p", "preview", "Preview the sample for test", cmd, false);
        verbose=std::make_shared<TCLAP::SwitchArg>("v", "verbose", "Display the verbose log messages", cmd, false);
        height=std::make_shared<TCLAP::ValueArg<int>>("H", "height", "Scaled sample height", false, 0, "scaled_sample_height", cmd);
        width=std::make_shared<TCLAP::ValueArg<int>>("W", "width", "Scaled sample width", false, 0, "scaled_sample_width", cmd);
        binarizedType=std::make_shared<TCLAP::ValueArg<int>>("b", "binarize", "Binarise the sample image", false, 0, "binarise_image", cmd);
        edgeWidth=std::make_shared<TCLAP::ValueArg<int>>("e", "edge", "Remove edge before applying PCA", false, 0, "edge_width_to_remove", cmd);
        grayscaled=std::make_shared<TCLAP::ValueArg<int>>("g", "grayscale", "Grayscale image before applying PCA", false, 1, "load_images_in_grayscale", cmd);
        contrastStretched=std::make_shared<TCLAP::ValueArg<int>>("c", "contrast", "Stretching contrast before applying grayscaling & PCA", false, 0, "contrast_stretch", cmd);
        reservedVariance=std::make_shared<TCLAP::ValueArg<float>>("V", "variance", "Reserved variance, ranging in (0,1.0], but 0.95~0.99 is normallly a good option", false, DEFAULT_RESERVED_VARIANCE, "reserved_variance", cmd);
        classTo=std::make_shared<TCLAP::ValueArg<int>>("t", "to_class_label", "Classes label id to which to train", true, 0, "class_to", cmd);
        classFrom=std::make_shared<TCLAP::ValueArg<int>>("f", "from_class_label", "Classes label id from which to train", true, 0, "class_from", cmd);

        cmd.parse(argc, argv);
    }catch(TCLAP::ArgException &e){
        LOGL(ERROR)<<e.error().c_str()<<" for arg " << e.argId();
        return false;
    }


    param->classFrom=classFrom->getValue();
    param->classTo=classTo->getValue();
    param->scaledWidth=width->getValue();
    param->scaledHeight=height->getValue();
    param->binarizedType=binarizedType->getValue();
    param->grayscaled= grayscaled->getValue() ? true : false;
    param->contrastStretched = contrastStretched->getValue() ? true : false;
    param->reservedVariance=reservedVariance->getValue();
    param->trainSamplesPath=trainSamplesPath->getValue();
    param->testSamplesPath=testSamplesPath->getValue();
    param->pcaPath=pcaPath->getValue();
    param->edgeWidth=edgeWidth->getValue();
    param->preview=preview->getValue();
    param->verbose=verbose->getValue();

    // 设置Log级别
    if(param->verbose){
        art::setLogLevel(art::LogLevel::INFO);
    }else{
        art::setLogLevel(art::LogLevel::NOTICE);
    }


    if(param->classTo - param->classFrom < 2 ){
        LOGL(ERROR)<<"Classes' count should be larger than 2 !";
       return false;
    }

    if(param->reservedVariance<=0.0 || param->reservedVariance>1.0 ){
        LOGL(ERROR)<<"Reserved variance value should be ranged in ( 0, 1.0 ]";
        return false;
    }

    LOG()<<"Class label id range: [ "<< param->classFrom << "," << param->classTo<<" ]";
    LOG()<<"Reserved variance: "<< param->reservedVariance;
    LOG()<<"Train-samples path: "<< param->trainSamplesPath;
    LOG()<<"Test-samples path: "<< param->testSamplesPath;
    LOG()<<"PCA model path: "<< param->pcaPath;

    return true;
}

/** 获取指定目录下的文件列表。需要libboost支持。
 * @param dir 指定目录
 * @param recursive 是否递归寻找
 * @param needSort 对列表是否进行排序
 * @filenames 保存的文件列表结果
 */
void searchFiles(const std::string dir, std::vector<std::string>& filenames, bool recursive=false, bool needSort=false){
    auto sortByName = [](std::string name1, std::string name2){
        return name1 < name2;
    };

    fs::path path(dir);

    if (!fs::exists(path)){
        DLOGL(ERROR, "Invalid search path!");
        return;
    }

    // Converts path, which must exist, to an absolute path that has no symbolic link, dot, or dot-dot elements.
    fs::path canonicaled_path=fs::canonical(path);
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(canonicaled_path); iter!=end_iter; ++iter){
        if (fs::is_regular_file(iter->status())){
            filenames.push_back(iter->path().string());
        }

        if (recursive && fs::is_directory(iter->status())){
            searchFiles(iter->path().string(), filenames, needSort, recursive);
        }
    }

    if(needSort){
        std::sort(filenames.begin(), filenames.end(), sortByName);
    }

}

/** 对比度拉伸
 *
 *
 */
cv::Mat contrastStretching(const cv::Mat& inputImage, double minVal, double maxVal)
{
    // 保存结果图像
	cv::Mat result;
    // 颜色查找表；利用颜色查找表效率高
	cv::Mat lut(1,256,CV_8U);
	for (int i=0; i<256; i++) {
        double temp=(i-minVal)/(maxVal-minVal)*255+0.5;
        if(temp>255) temp=255;
        if(temp<0) temp=0;
        lut.at<uchar>(i)=temp;
	}

    if(inputImage.channels()>=3){
        cv::Mat gray;
        cv::cvtColor(inputImage, gray, CV_BGR2GRAY);
        cv::LUT(gray, lut, result);
    }else if(inputImage.channels()==1){
        cv::LUT(inputImage, lut, result);
    }else{
        result=cv::Mat();
    }

    return result;
}


int processSamples(std::string path, std::vector<cv::Mat>& processedSamples){
    std::vector<std::string> sampleFiles;
    cv::Mat img;
    cv::Mat preview;
    int previewCount=0;

    if (!fs::exists(fs::path(path))){
        LOGL(ERROR)<<"Sample path ("<<path<<") not exist!";
        return 0;
    }

    searchFiles(path, sampleFiles);


    for(auto& f : sampleFiles){
        if(g_param.grayscaled){
            //// 大多数颜色无关的应用中，都是用灰度图像即可：
            img=cv::imread(f, cv::IMREAD_GRAYSCALE );
        }else{
            //// 如果希望原封不动(如：不自动去除alpha通道)地把图片信息作为样本，则如下读取：
            img=cv::imread(f, cv::IMREAD_UNCHANGED);
        }
        if(!img.data){
            LOGL(ERROR)<<"Failed to read image.";
            continue;
        }

        //// 检查是否缩放
        if(g_param.scaledHeight>0 && g_param.scaledWidth>0){
            cv::Mat temp(cv::Size(g_param.scaledWidth, g_param.scaledHeight), img.type());
            cv::resize(img, temp, temp.size(), cv::INTER_AREA); // generally, it's a shrink op
            img=temp;
        }

        //// 检查是否要移除边缘
        if(g_param.edgeWidth>0){
            cv::Mat temp=img(cv::Rect(g_param.edgeWidth,g_param.edgeWidth, img.cols-2*g_param.edgeWidth, img.rows-2*g_param.edgeWidth));
            img=temp;
        }

        //// 注意，进行简单的预处理未必能够提升PCA的特征抽取能力；相反，可能降低！
        //// 因为预处理可能把具有区分度的特征抛弃掉！

        //// 下面是对样本图像进行对比度拉伸。
        if(g_param.contrastStretched){
            double minVal, maxVal;
            cv::minMaxLoc(img, &minVal, &maxVal);
            cv::Mat temp=contrastStretching(img, minVal, maxVal);
            img=temp;
        }

        //// 下面是另一个简单的预处理：
        //// 把灰度图阈值化转为黑白两色图像。实验证明，阈值化后的图像PCA降维
        if(img.type()==CV_8UC1){
            if(g_param.binarizedType==1){
                cv::threshold(img, img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
            }else if(g_param.binarizedType==2){
                cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 9);
            }
        }

        if(g_param.preview){
            if(preview.data==nullptr){
                preview=cv::Mat(cv::Size(img.cols*g_param.previewWidth, img.rows*g_param.previewHeight), img.type(), cv::Scalar(0));
            }

            cv::Mat roi=preview(cv::Rect(img.cols*(previewCount%g_param.previewWidth), img.rows*(previewCount/g_param.previewWidth), img.cols, img.rows));
            img.copyTo(roi);
            previewCount++;
            //cv::imshow("roi", roi);
            //cv::waitKey(0);
            if( previewCount>=g_param.previewWidth*g_param.previewHeight ){
                cv::imshow("pewview", preview);
                cv::waitKey(0);
                previewCount=0;
                preview.release();
            }

        }

        // 保存经过预处理的样本
        processedSamples.push_back(img);
    }


    if(g_param.preview && previewCount>0){
        cv::imshow("pewview", preview);
        cv::waitKey(0);
        previewCount=0;
        preview.release();
    }

    return sampleFiles.size();

}


bool loadSamples(std::vector<cv::Mat>& trainSamples, std::vector<cv::Mat>& testSamples, std::vector<cv::Mat>& allSamples){
    int n;
    Parameters* param=&g_param;

    trainSamples.clear();
    testSamples.clear();
    allSamples.clear();

    for(int i=param->classFrom; i<=param->classTo; i++){
        std::string path;

        path=param->trainSamplesPath+"/"+std::to_string(i);
        n=processSamples(path, trainSamples);
        if(!n){
            LOGL(ERROR)<<"Failed to load train sample files!";
            return false;
        }else{
            LOGL(INFO)<<"Class-"<<i<<" "<<n<<" training-samples loaded.";
        }
        param->trainSamplesCountInClasses.push_back(n);

        path=param->testSamplesPath+"/"+std::to_string(i);
        n=processSamples(path, testSamples);
        if(!n){
            LOGL(ERROR)<<"Failed to load test sample files!";
            return false;
        }else{
            LOGL(INFO)<<"Class-"<<i<<" "<<n<<" testing-samples loaded.";
        }
        param->testSamplesCountInClasses.push_back(n);
    }

    // 把训练集和测试集汇总到一起
    for(auto & e : trainSamples){
        allSamples.push_back(e);
    }
    for(auto& e: testSamples){
        allSamples.push_back(e);
    }

    return true;

}


    /// 把cv::Mat的列表转为行矩阵形式(row-matrix), 类型目前为CV_32FC1
    cv::Mat convertToRowMatrix(std::vector<cv::Mat>& src){
        // Number of samples:
        size_t n = src.size();
        // Return empty matrix if no matrices given:
        if(n == 0) return cv::Mat();
        // dimensionality of (reshaped) samples
        size_t d = src[0].total();

        // Create resulting data matrix:
        cv::Mat data(n, d, CV_32FC1);
     

        // Now copy data:
        for(int i = 0; i < n; i++) {
            //
            if(src[i].empty()) {
                std::cerr<<"Image("<<i<<") is invalid (empty image)!\n";
                assert(0);
            }
            // Make sure data can be reshaped, throw a meaningful exception if not!
            if(src[i].total() != d) {
                std::cerr<<"Image("<<i<<") has wrong number of elements!\n";
                assert(0);                
            }
            // Get a hold of the current row:
            cv::Mat rmat = data.row(i);

            // 把原始矩阵转为行矩阵
            src[i].clone().reshape(1, 1).convertTo(rmat, rmat.type());
    
        }

        return data;
    }

bool generatePCAModel(std::vector<cv::Mat>& samples, cv::PCA& pca){
    Parameters* param=&g_param;

    // 把全部样本集中到一个矩阵cv::Mat中，每一行一个样本
    cv::Mat samplesRowMatrix=convertToRowMatrix(samples);
    LOGL(NOTICE)<<"Running PCA dimension reducing algorithm on "<<samples.size()<<" samples, please wait ...";
    pca(samplesRowMatrix, cv::Mat(), cv::PCA::DATA_AS_ROW, param->reservedVariance);
    LOG()<<"Done.";

    if(pca.eigenvalues.empty() || pca.eigenvectors.empty() || samples[0].total()!=pca.eigenvectors.cols){
        LOGL(ERROR)<<"Failed to generate PCA model!";
        return false;
    }

    return true;
}


void createLabels(std::vector<int>& classSamplesCount, std::vector<int>& labels){
    Parameters* param=&g_param;
    int label=param->classFrom;

    labels.clear();
    for(int n: classSamplesCount){
        for(int i=0; i<n; i++){
            labels.push_back(label);
        }
        label++;
    }
}


bool generateLIBSVMData(std::string filename,  std::vector<cv::Mat>& samples, std::vector<int>& labels,  cv::PCA& pca){
    std::fstream file;
    std::stringstream ss;

    // 打开文件准备存储
    file.open(filename, std::ios::out);
    if(!file.is_open()){
        LOGL(ERROR)<<"Failed to write data file!";
        return false;
    }

    // 对样本进行降维
    cv::Mat rowMatrix=convertToRowMatrix(samples);
    cv::Mat compressed;
    pca.project(rowMatrix, compressed);

    if(compressed.rows != samples.size() || compressed.cols!= pca.eigenvectors.rows){
        LOGL(ERROR)<<"Failed to do PCA dimesion-reduction on samples!";
        return false;
    }

    // 保存样本到数据文件
    for(int r=0; r<compressed.rows; r++){
        ss.clear();
        ss.str("");

        ss<<labels[r]<<" ";
        cv::Mat mat=compressed.row(r);
        for(int c=0; c<mat.cols; c++){
            ss<<c+1<<":"<<mat.at<float>(0,c)<<" ";
        }
        file<<ss.str()<<"\n";
    }


    file.close();

}



int main(int argc, char** argv){
    std::vector<cv::Mat> trainSamples; ///< 训练样本集
    std::vector<cv::Mat> testSamples; ///< 测试样本集
    std::vector<cv::Mat> allSamples;  ///< 总样本集：就是上述训练集和测试集的总和
    cv::PCA pca; ///< OpenCV自带的PCA算法

    // 解析命令行参数
    if(!parseCommandLine(argc, argv)){
        LOGL(ERROR)<<"Parsing command line error!";
        return -1;
    }

    // 载入样本, 每个样本都是一个图像cv::Mat
    if(!loadSamples(trainSamples, testSamples, allSamples)){
        LOGL(ERROR)<<"Failed to load samples!";
        return -1;
    }

    if(!trainSamples.size()){
        LOGL(ERROR)<<"Train-samples set is empty!";
        return -1;
    }else{
        LOG()<<"Train-samples count: "<<trainSamples.size();
    }

    if(!testSamples.size()){
        LOGL(ERROR)<<"Test-samples set is empty!";
        return -1;
    }else{
        LOG()<<"Test-samples count: "<<testSamples.size();
    }

    // 从总样本集计算PCA模型
    if(!generatePCAModel(allSamples, pca)){
        LOGL(ERROR)<<"Failed to generate PCA model!";
        return -1;
    }else{
        LOG()<<"PCA model generated.";
    }

    /// 保存计算出来的PCA对象
    std::string pcaFile=g_param.pcaPath+"/pca_c"+std::to_string(g_param.classFrom)+"-"+std::to_string(g_param.classTo)+"v"+std::to_string((int)(g_param.reservedVariance*100));
    if(g_param.edgeWidth>0){
        pcaFile+="_noEdge";
    }
    pcaFile+=".xml";
    cv::FileStorage storage(pcaFile, cv::FileStorage::WRITE);
    pca.write(storage);
    storage.release();
    LOGL(NOTICE)<<"PCA model saved to "<<pcaFile;
    LOGL(NOTICE)<<"Samples' dimension could be reduced to " << pca.eigenvalues.rows <<" when reserving "<<g_param.reservedVariance*100<<"% variance.";

    // 为训练集和测试集生成label
    std::vector<int> trainLabels, testLabels;
    createLabels(g_param.trainSamplesCountInClasses, trainLabels);
    createLabels(g_param.testSamplesCountInClasses, testLabels);

    // 生成训练集和测试集文件（LIBSVM使用的数据格式）
    LOG()<<"Saving train & test samples into LIBSVM data format...";
    bool ret;
    std::string trainSamplesFile=g_param.trainSamplesPath+"/train_c"+std::to_string(g_param.classFrom)+"-"+std::to_string(g_param.classTo)+"v"+std::to_string((int)(g_param.reservedVariance*100));
    if(g_param.edgeWidth>0){
        trainSamplesFile+="_noEdge";
    }
    trainSamplesFile+=".dat";
    ret=generateLIBSVMData(trainSamplesFile, trainSamples, trainLabels, pca);
    if(ret){
        LOGL(NOTICE)<<"Train-samples data file saved to "<<trainSamplesFile;
    }else{
        return -1;
    }

    std::string testSamplesFile=g_param.testSamplesPath+"/test_c"+std::to_string(g_param.classFrom)+"-"+std::to_string(g_param.classTo)+"v"+std::to_string((int)(g_param.reservedVariance*100));
    if(g_param.edgeWidth>0){
        testSamplesFile+="_noEdge";
    }
    testSamplesFile+=".dat";
    ret=generateLIBSVMData(testSamplesFile, testSamples, testLabels, pca);
    if(ret){
        LOGL(NOTICE)<<"Test-samples data file saved to "<<testSamplesFile;
    }else{
        return -1;
    }

    return 0;
}
