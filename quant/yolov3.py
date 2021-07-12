import sys
import os
import cv2
import numpy as np
import argparse
import caffe
import amct_caffe as amct


def mkdir(name):
    if not os.access(name, os.F_OK):
        os.makedirs(name)


def img_preprocess(img):
    img = cv2.resize(img, (416, 416)) 
    img = img.swapaxes(0, 2)
    img = img.swapaxes(1, 2)
    offset = [104, 117, 123]
    ch = 3
    for c in range(ch):
        img[c, ...] -= offset[c]
    img = img.astype("float32") / 255
    img = img.reshape(1, 3, 416, 416)
    return img


def calibration(net, videos, iter_num):
    for video in videos:
        calibration_one_video(net, video, iter_num)


def calibration_one_video(net, video_file, iter_num):
    cap = cv2.VideoCapture(video_file)

    info = np.zeros((1, 4), dtype="float32")
    info[...] = 416

    iter_i = 0

    while cap.isOpened():
        if iter_i >= iter_num:
            break
        ret, frame = cap.read()
        if ret:
            img = img_preprocess(frame)
            feed_dict = {"data": img, "img_info": info}
            net.forward(**feed_dict)
            iter_i += 1
            print("calibrate video {} frame {}/{}".format(video_file, iter_i, iter_num))


def main():
    parser = argparse.ArgumentParser(description='Example of convert yolov3 caffe model to int8')
    parser.add_argument('--prototxt', type=str, help='path to yolov3_pp.prototxt', default='model/yolov3_pp.prototxt')
    parser.add_argument('--caffemodel', type=str, help='path to yolov3.caffemodel', default='model/yolov3.caffemodel')
    parser.add_argument('--tmp_dir', type=str, help='path to save temp files', default='./tmp')
    parser.add_argument('--output_dir', type=str, help='path to save output model file',
                        default='model/')
    parser.add_argument('--output_model_name', type=str, help='prefix of output model files',
                        default='yolov3_int8')
    parser.add_argument('--calib_video', action='append', required=True,
                        help='videos used in calibration, it can be specified multiple times.')
    parser.add_argument('--calib_frame', type=int,
                        help='number of frames used in each video file',
                        default=200)
    args = parser.parse_args()
    tmp_dir = args.tmp_dir

    mkdir(tmp_dir)
    caffe.set_mode_cpu()

    config_json_file = os.path.join(tmp_dir, 'config.json')
    skip_layers = []
    batch_num = 1

    amct.create_quant_config(config_json_file,
                             args.prototxt,
                             args.caffemodel,
                             skip_layers,
                             batch_num)
    
    scale_offset_record_file = os.path.join(tmp_dir, 'scale_offset_record.txt')
    graph = amct.init(config_json_file,
                      args.prototxt,
                      args.caffemodel,
                      scale_offset_record_file)
    print("done init")
    modified_model_file = os.path.join(tmp_dir, 'modified_model.prototxt')
    modified_weights_file = os.path.join(tmp_dir, 'modified_model.caffemodel')
    
    amct.quantize_model(graph, modified_model_file, modified_weights_file)
    print("done quantize")

    net = caffe.Net(modified_model_file, modified_weights_file, caffe.TEST)
    calibration(net, args.calib_video, args.calib_frame)

    result_path = os.path.join(args.output_dir, args.output_model_name)
    amct.save_model(graph, 'Both', result_path)
    print("done save")


if __name__ == '__main__':
    main()
