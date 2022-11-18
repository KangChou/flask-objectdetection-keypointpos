import os
import cv2
from base_camera import BaseCamera
from PicoDet import PicoDet
net = PicoDet(model_pb_path='models/picodet_s_320_coco.onnx', label_path='models/models.names', prob_threshold=0.4, iou_threshold=0.3)

from yoloenet import PP_YOLOE,KeyPointDetector
net = PP_YOLOE(prob_threshold=0.7)
kpt_predictor = KeyPointDetector()



class Camera(BaseCamera):

    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, srcimg = camera.read()

            # srcimg = cv2.imread(args.imgpath)
            # 算法：人体姿态关键点检测
            dets = net.detect(srcimg)
            for i in range(dets.shape[0]):
                xmin, ymin, xmax, ymax = int(dets[i, 0]), int(dets[i, 1]), int(dets[i, 2]), int(dets[i, 3])
                results = kpt_predictor.predict(srcimg[ymin:ymax, xmin:xmax],
                                                {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
                srcimg = kpt_predictor.visualize_pose(srcimg, results)

            # 算法：目标检测
            # img = net.detect(srcimg)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', srcimg)[1].tobytes()
