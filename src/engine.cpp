#include "engine.hpp"

InferEngine::InferEngine()
{
}

InferEngine::~InferEngine() {}

void InferEngine::loadNetwork(const string network_name, size_t input_size)
{
	Core ie;
	auto network = ie.ReadNetwork(network_name + ".xml");
	network.setBatchSize(1);

	InputsDataMap input_info(network.getInputsInfo());

    if (input_info.size() != input_size) {
        throw std::logic_error("Network input size not match");
	}
	InputInfo::Ptr input_data = input_info.begin()->second;
	if (!input_data) {
		throw std::runtime_error("Input data pointer is invalid");
	}
    input_data->setPrecision(Precision::U8);
	input_name = input_info.begin()->first;

	/** Taking information about all topology outputs **/
    OutputsDataMap output_info(network.getOutputsInfo());
	DataPtr output_data = output_info.begin()->second;
	if (!output_data) {
		throw std::runtime_error("Output data pointer is invalid");
	}
	output_data->setPrecision(Precision::FP32);
	output_name = output_info.begin()->first;

	ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
	request = executable_network.CreateInferRequestPtr();
}

void InferEngine::submitRequest()
{
	if (!request)
		return;
    request->StartAsync();
}

void InferEngine::wait()
{
	if (!request)
		return;
	request->Wait(IInferRequest::WaitMode::RESULT_READY);
}

void FaceDetect::loadNetwork(const string network_name, size_t input_size)
{	
	InferEngine::loadNetwork(network_name, input_size);
	
	Blob::Ptr output_data = request->GetBlob(output_name);
	const SizeVector outputDims = output_data->getTensorDesc().getDims();
	maxProposalCount = outputDims[2];
	objectSize = outputDims[3];
}

void FaceDetect::enqueue(const Mat &frame)
{
	width  = static_cast<size_t>(frame.cols);
	height = static_cast<size_t>(frame.rows);

	/** Getting input blob **/
	Blob::Ptr input = request->GetBlob(input_name);
	auto input_buffer = input->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();

	/** Fill input tensor with planes. First b channel, then g and r channels **/
	if (frame.empty()) throw logic_error("Invalid frame");
	auto dims = input->getTensorDesc().getDims();
	size_t channels_number = dims[1];
	size_t frame_size = dims[3] * dims[2];
	Mat resized_frame(frame);
	/* Resize and copy data from the frame to the input blob */
	resize(frame, resized_frame, cv::Size(dims[3], dims[2]));
	for (size_t pid = 0; pid < frame_size; ++pid) {
    	for (size_t ch = 0; ch < channels_number; ++ch) {
        	input_buffer[ch * frame_size + pid] = resized_frame.at<cv::Vec3b>(pid)[ch];
    	}
	}
}

vector<Rect> FaceDetect::fetchResults()
{
	const float *detections = request->GetBlob(output_name)->buffer().as<float *>();

	vector<Rect> results;
	for (int i = 0; i < maxProposalCount; i++) {
		//float image_id = detections[i * objectSize + 0];
		//int label = static_cast<int>(detections[i * objectSize + 1]);
		float confidence = detections[i * objectSize + 2];
		int face_x = static_cast<int>(detections[i * objectSize + 3] * width);
		int face_y = static_cast<int>(detections[i * objectSize + 4] * height);
		int face_width = static_cast<int>(detections[i * objectSize + 5] * width - face_x);
		int face_height = static_cast<int>(detections[i * objectSize + 6] * height - face_y);

		if (confidence <= 0.5) continue;

#if 0
		int center_x = face_x + face_width/2;
		int center_y = face_y + face_height/2;
		int max_of_size = max(face_width, face_height);
		int new_face_width = static_cast<int>(1.2 * max_of_size);
		int new_face_height = static_cast<int>(1.2 * max_of_size);
		
		Rect roi;
		roi.x = center_x - new_face_width/2;
		roi.y = center_y - new_face_height/2;
		roi.width = new_face_width;
		roi.height = new_face_height;
#else
		Rect roi;
		roi.x = face_x;
		roi.y = face_y;
		roi.width = face_width;
		roi.height = face_height;
#endif

		results.push_back(roi);
	}
	return results;
}

void LandmarkDetect::submitRequest()
{
    InferEngine::submitRequest();
    enquedFaces = 0;
}

void LandmarkDetect::enqueue(const Mat &frame)
{
	/** Getting input blob **/
	Blob::Ptr input = request->GetBlob(input_name);
	auto input_buffer = input->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();

	/** Fill input tensor with planes. First b channel, then g and r channels **/
	if (frame.empty()) throw logic_error("Invalid frame");
	auto dims = input->getTensorDesc().getDims();
	size_t channels_number = dims[1];
	size_t frame_size = dims[3] * dims[2];
	Mat resized_frame(frame);
    size_t batchOffset = enquedFaces * frame_size * channels_number;

	/* Resize and copy data from the frame to the input blob */
	resize(frame, resized_frame, cv::Size(dims[3], dims[2]));
	for (size_t pid = 0; pid < frame_size; ++pid) {
    	for (size_t ch = 0; ch < channels_number; ++ch) {
        	input_buffer[batchOffset + ch * frame_size + pid] = resized_frame.at<cv::Vec3b>(pid)[ch];
    	}
	}

	enquedFaces++;
}

vector<float> LandmarkDetect::operator[] (int idx) const
{
	auto landmarksBlob = request->GetBlob(output_name);
    auto n_lm = landmarksBlob->getTensorDesc().getDims()[1];
    const float *normed_coordinates = request->GetBlob(output_name)->buffer().as<float *>();
	vector<float> normedLandmarks;

    auto begin = n_lm * idx;
    auto end = begin + n_lm / 2;
	for (auto i_lm = begin; i_lm < end; ++i_lm) {
		float normed_x = normed_coordinates[2 * i_lm];
		float normed_y = normed_coordinates[2 * i_lm + 1];

		//cout << normed_x << ", " << normed_y << std::endl;
        normedLandmarks.push_back(normed_x);
        normedLandmarks.push_back(normed_y);
    }
	return normedLandmarks;
}
