#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <gflags/gflags.h>
#include "engine.hpp"
#include "mqtt.h"
#include <syslog.h>

using namespace std;
using namespace cv;

#define UNKNOWN_PERSON_ID -10

void AlignFaces(vector<Mat>* face_images,
                vector<Mat>* landmarks_vec);

struct face_db_t {
	int personID;
	vector<float> features;
};

struct seen_face_t {
	int count;
	int id;
};

volatile bool running = true;
volatile bool performRegistration = false;

const char* DEFAULT_DB_LOCATION = "./defaultdb.bin";
const char* DEFAULT_THUMBNAIL_PATH = "./thumbs/";
const char* DEFAULT_MODELS_PATH = "./models/";
const int FONT = cv::FONT_HERSHEY_PLAIN;
const Scalar GREEN(0, 255, 0);
const Scalar BLUE(255, 0, 0);
const Scalar WHITE(255, 255, 255);

Mat frame;
VideoCapture webcam;

FaceDetect face_detect;
LandmarkDetect landmark_detect;
LandmarkDetect image_reid;

vector<Rect> detectedFaces;
vector<int>  personIDs;
vector<face_db_t> face_db;
vector<seen_face_t> seen_face_db;

int cameraNumber = 0;
string databaseLocation;
string thumbnailPath;
string modelsPath;

string lastTopic;
string lastID;

bool loadFaceDB(string filepath)
{
	ifstream file(filepath);
    if (!file.is_open()) {
        syslog(LOG_WARNING, "Unable to locate face DB. Will be created on save.");
        return false;
    }
	face_db_t new_face;
	new_face.features = vector<float> (256);

	face_db.clear();
	while(file.peek() != EOF) {
		file.read((char *)&new_face.personID, sizeof(int));
		file.read((char *)&new_face.features[0], new_face.features.size()*sizeof(float));
		face_db.push_back(new_face);
	}
	
	file.close();

	return true;
}

bool saveFaceDB(string filepath)
{
	ofstream file(filepath);
	for (auto && face : face_db) {
		file.write((char*)&face.personID, sizeof(int));;
		file.write((char*)&face.features[0], face.features.size() * sizeof(float));
	}
	file.close();

	return true;
}

#if 0
void remove_face_data(string name)
{
	ifstream infile("faceData.bin");
	vector<face_data_t> data;
	face_data_t new_face;
	new_face.landmark = vector<float> (70);

	while(infile.peek() != EOF) {
		getline(infile, new_face.name);
		infile.read((char *)&new_face.landmark[0], sizeof(float)*70);
		if (new_face.name == name)
			continue;
		data.push_back(new_face);
	}	
	infile.close();

	ofstream outfile;
	outfile.open("faceData.bin");
	for(size_t i=0;i<data.size();i++) {
		outfile << data[i].name << endl;
		outfile.write((char*)&data[i].landmark[0], data[i].landmark.size() * sizeof(float));
	}
	outfile.close();
}
#endif

void updateSeenFaces()
{
	for(int i=personIDs.size()-1;i>-1;i--) {
		int match=0;
		for(size_t j=0;j<seen_face_db.size();j++) {
			if(personIDs[i] == seen_face_db[j].id) {
				seen_face_db[j].count = 30;
				personIDs.erase(personIDs.begin()+i);
				match=1;
				break;
			}
		}
		if(!match) {
			seen_face_t new_seen_face;
			new_seen_face.id = personIDs[i];
			new_seen_face.count = 30;
			seen_face_db.push_back(new_seen_face);
		}
	}
	for(int i=seen_face_db.size()-1;i>-1;i--) {
		seen_face_db[i].count--;
		if(seen_face_db[i].count == 0)
			seen_face_db.erase(seen_face_db.begin()+i);
	}
}

float ComputDistance(const vector<float> descr1, const vector<float> descr2)
{
	assert(descr1.size() == descr2.size());
	float xy = 0;
	float xx = 0;
	float yy = 0;
	for(uint32_t i=0;i<descr1.size();i++) {
		xy += descr1[i] * descr2[i];
		xx += descr1[i] * descr1[i];
		yy += descr2[i] * descr2[i];
	}
    float norm = sqrt(xx * yy) + 1e-6f;
    return 1.0f - xy / norm;
}

int createNewPersonID()
{
	int maxID = 0;
	for(auto && face : face_db) {
		if (face.personID > maxID)
			maxID = face.personID;
	}
	return maxID+1;
}

void parseArgs(int argc, const char* argv[]) {
	if (argc > 1) {
    	cameraNumber = atoi(argv[1]);
	}

	databaseLocation = std_getenv("FACE_DB");
	if (databaseLocation.empty()) {
    	databaseLocation = DEFAULT_DB_LOCATION;
	}

	thumbnailPath = std_getenv("FACE_IMAGES");
	if (thumbnailPath.empty()) {
    	thumbnailPath = DEFAULT_THUMBNAIL_PATH;
	}
	
	modelsPath = std_getenv("MODELS");
	if (modelsPath.empty()) {
    	modelsPath = DEFAULT_MODELS_PATH;
	}
}

// publish MQTT message with a JSON payload
void publishMQTTMessage(const string& topic, const string& id)
{
	// don't send repeat messages
	if (lastTopic == topic && lastID == id) {
    	return;
	}

	lastTopic = topic;
	lastID = id;

	string payload = "{\"id\": \"" + id + "\"}";

	mqtt_publish(topic, payload);

	string msg = "MQTT message published to topic: " + topic;
	syslog(LOG_INFO, "%s", msg.c_str());
	syslog(LOG_INFO, "%s", payload.c_str());
}

// message handler for the MQTT subscription for the "commands/register" topic
int handleControlMessages(void *context, char *topicName, int topicLen, MQTTClient_message *message)
{
	string topic = topicName;
	string msg = "MQTT message received: " + topic;
	syslog(LOG_INFO, "%s", msg.c_str());

	if (topic == "commands/register") {
    	performRegistration = true;
	}
	return 1;
}

bool openWebcamInput(int cameraNumber)
{
	webcam.open(cameraNumber);

	if (!webcam.isOpened())
	{
		syslog(LOG_ERR, "Error: fail to capture video.");
		return false;
	}

	return true;
}

bool getNextImage()
{
	webcam >> frame;
	if (frame.empty())
	{
		syslog(LOG_ERR, "Error: no input image.");
		return false;
	}
	return true;
}

int initFaceDetection()
{
	face_detect.loadNetwork(modelsPath+"/face-detection-adas-0001");
	landmark_detect.loadNetwork(modelsPath+"/landmarks-regression-retail-0009");
	image_reid.loadNetwork(modelsPath+"/face-reidentification-retail-0095");
    
	loadFaceDB(databaseLocation);
	
	return true;
}

void lookForFaces()
{
    personIDs.clear();
	
	face_detect.enqueue(frame);
	face_detect.submitRequest();
	face_detect.wait();
	detectedFaces = face_detect.fetchResults();
}

vector<float> getFaceFeatures(Mat face)
{
	landmark_detect.enqueue(face);
	landmark_detect.submitRequest();
	landmark_detect.wait();
	vector<float> Landmarks = landmark_detect[0];
	Mat landmarks_mat(Landmarks.size()/2, 2, CV_32F);
	for (int i = 0; i < landmarks_mat.rows; i++) {
		landmarks_mat.at<float>(i, 0) = Landmarks[i*2];
		landmarks_mat.at<float>(i, 1) = Landmarks[i*2 + 1];
	}

	vector<Mat> images = {face};
	vector<Mat> landmarks_vec = {landmarks_mat};
	AlignFaces(&images, &landmarks_vec);

	image_reid.enqueue(face);
	image_reid.submitRequest();
	image_reid.wait();
	return image_reid[0];
}

void recognizeFaces()
{
	if (detectedFaces.size() > 0) {	
        bool saveNeeded = false;
		for (auto && rect : detectedFaces) {
			Rect roi = rect & Rect(0, 0, frame.cols, frame.rows);
		
			Mat crop;
			frame(roi).copyTo(crop);
		
			vector<float> features = getFaceFeatures(crop);

			if(!face_db.empty())
				for(auto && face : face_db) {
					if(ComputDistance(features, face.features) < 0.3)
						personIDs.push_back(face.personID);
					else
						personIDs.push_back(UNKNOWN_PERSON_ID);
				}
			else
				personIDs.push_back(UNKNOWN_PERSON_ID);
		}

		if (!performRegistration)
			updateSeenFaces();

	    for (uint i = 0; i < personIDs.size(); i++)
		{
			if (personIDs[i] == UNKNOWN_PERSON_ID)
			{
				if (performRegistration)
				{
					//create new person ID
					face_db_t new_face;
					new_face.features = vector<float> (256);
                	new_face.personID = createNewPersonID();
					Rect roi = detectedFaces[i] & Rect(0, 0, frame.cols, frame.rows);
					Mat crop;
					frame(roi).copyTo(crop);
					new_face.features = getFaceFeatures(crop);
					face_db.push_back(new_face);

					publishMQTTMessage("person/registered", to_string(new_face.personID));

					string saveFileName = thumbnailPath+to_string(new_face.personID)+".jpg";
					imwrite(saveFileName.c_str(), frame);

					saveNeeded = true;
					performRegistration = false;
				} else {
					publishMQTTMessage("person/seen", "UNKNOWN");
				}
			} else {
				publishMQTTMessage("person/seen", to_string(personIDs[i]));
			}
		}
		if (saveNeeded)
			saveFaceDB(databaseLocation);
	} else {
		lastTopic.clear();
		lastID.clear();
	}
}

// display the info on any faces detected in the window image
void displayDetectionInfo()
{
	for (uint i = 0; i < detectedFaces.size(); ++i)
	{
		Rect faceRect = detectedFaces[i] & Rect(0, 0, frame.cols, frame.rows);

		// Draw face rect
		rectangle(frame, faceRect, WHITE, 2);
	}
}

// display the info on any faces recognized in the window image
void displayRecognitionInfo()
{
    String str;

	for (uint i = 0; i < detectedFaces.size(); i++)
	{
		Rect faceRect = detectedFaces[i] & Rect(0, 0, frame.cols, frame.rows);

		//draw FR info
		str = (personIDs[i] > 0) ? format("Person: %d", personIDs[i]) : "UNKNOWN";

		Size strSize = cv::getTextSize(str, FONT, 1.2, 2, NULL);
		Point strPos(faceRect.x + (faceRect.width / 2) - (strSize.width / 2), faceRect.y - 2);
		putText(frame, str, strPos, FONT, 1.2, GREEN, 2);
	}
}

// Output BGR24 raw format to console.
void outputFrame() {
	Vec3b pixel;
	for(int j = 0;j < frame.rows;j++){
		for(int i = 0;i < frame.cols;i++){
			pixel = frame.at<Vec3b>(j, i);
			printf("%c%c%c", pixel[0], pixel[1], pixel[2]);
  		}
	}
	fflush(stdout);
}

// display the window image
void display() {
	displayDetectionInfo();
	displayRecognitionInfo();

	outputFrame();
}

int main(int argc, const char* argv[])
{
	syslog(LOG_INFO, "Starting cvservice...");

    parseArgs(argc, argv);

    try
    {
		int result = mqtt_start(handleControlMessages);
		if (result == 0) {
			syslog(LOG_INFO, "MQTT started.");
		} else {
			syslog(LOG_INFO, "MQTT NOT started: have you set the ENV varables?");
		}

		mqtt_connect();
		mqtt_subscribe("commands/register");

		if (!openWebcamInput(cameraNumber)) {
			throw invalid_argument("Invalid camera number or unable to open camera device.");
			return 1;
		}

		if (!initFaceDetection()) {
			throw runtime_error("Unable to initialize face detection or face recognition.");
			return 1;
		}

		while(running)
		{
			if (getNextImage())
			{
				lookForFaces();

				recognizeFaces();

				display();
			}
		}

		mqtt_disconnect();
		mqtt_close();
		return 0;
	}
	catch(const std::exception& error)
	{
		syslog(LOG_ERR, "%s", error.what());
		return 1;
	}
	catch(...)
	{
		syslog(LOG_ERR, "Unknown/internal exception ocurred");
		return 1;
	}
}
