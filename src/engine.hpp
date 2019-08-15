#include <string>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class InferEngine
{
protected:
	InferencePlugin plugin;
	CNNNetwork network;
	InferRequest::Ptr request;

	string input_name;
	string output_name;
    
public:
	InferEngine();
	virtual ~InferEngine(); 
	virtual void loadNetwork(const string network_name, size_t input_size=1);
    virtual void submitRequest();
	virtual void wait();
};

class FaceDetect : public InferEngine
{
private:
	int maxProposalCount;
    int objectSize;
	size_t width;
	size_t height;

public:
	void loadNetwork(const string network_name, size_t input_size=1) override;
    void enqueue(const Mat &frame);
    vector<Rect> fetchResults();
};

class LandmarkDetect : public InferEngine
{
private:
    size_t enquedFaces = 0;
public:
    void submitRequest() override;
    void enqueue(const Mat &frame);
    vector<float> operator[] (int idx) const;
	Mat Compute(Size outp_shape);
};
