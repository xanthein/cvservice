// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

static const float h = 112.;
static const float w = 96.;
// reference landmarks points in the unit square [0,1]x[0,1]
static const float ref_landmarks_normalized[] = {
    30.2946f / w, 51.6963f / h, 65.5318f / w, 51.5014f / h, 48.0252f / w,
    71.7366f / h, 33.5493f / w, 92.3655f / h, 62.7299f / w, 92.2041f / h};

Mat GetTransform(Mat* src, Mat* dst) {
    Mat col_mean_src;
    reduce(*src, col_mean_src, 0, REDUCE_AVG);
    for (int i = 0; i < src->rows; i++) {
        src->row(i) -= col_mean_src;
    }

    Mat col_mean_dst;
    reduce(*dst, col_mean_dst, 0, REDUCE_AVG);
    for (int i = 0; i < dst->rows; i++) {
        dst->row(i) -= col_mean_dst;
    }

    Scalar mean, dev_src, dev_dst;
    meanStdDev(*src, mean, dev_src);
    dev_src(0) =
            max(static_cast<double>(numeric_limits<float>::epsilon()), dev_src(0));
    *src /= dev_src(0);
    meanStdDev(*dst, mean, dev_dst);
    dev_dst(0) =
            max(static_cast<double>(numeric_limits<float>::epsilon()), dev_dst(0));
    *dst /= dev_dst(0);

    Mat w, u, vt;
    SVD::compute((*src).t() * (*dst), w, u, vt);
    Mat r = (u * vt).t();
    Mat m(2, 3, CV_32F);
    m.colRange(0, 2) = r * (dev_dst(0) / dev_src(0));
    m.col(2) = (col_mean_dst.t() - m.colRange(0, 2) * col_mean_src.t());
    return m;
}

void AlignFaces(vector<Mat>* face_images,
                vector<Mat>* landmarks_vec) {
    if (landmarks_vec->size() == 0) {
        return;
    }
    CV_Assert(face_images->size() == landmarks_vec->size());
    Mat ref_landmarks = Mat(5, 2, CV_32F);

    for (size_t j = 0; j < face_images->size(); j++) {
        for (int i = 0; i < ref_landmarks.rows; i++) {
            ref_landmarks.at<float>(i, 0) =
                    ref_landmarks_normalized[2 * i] * face_images->at(j).cols;
            ref_landmarks.at<float>(i, 1) =
                    ref_landmarks_normalized[2 * i + 1] * face_images->at(j).rows;
			
            landmarks_vec->at(j).at<float>(i, 0) *= face_images->at(j).cols;
            landmarks_vec->at(j).at<float>(i, 1) *= face_images->at(j).rows;
        }
        Mat m = GetTransform(&ref_landmarks, &landmarks_vec->at(j));
        warpAffine(face_images->at(j), face_images->at(j), m,
                       face_images->at(j).size(), WARP_INVERSE_MAP);
    }
}
