import xml.etree.ElementTree as etree
from sklearn.metrics import average_precision_score
import os
import math

person_classes = ['person']
animal_classes = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
vehicle_classes = ['aeroplane', 'bicycle', 'boat', 'bus',
                   'car', 'motorbike', 'train']
indoor_classes = ['bottle', 'chair', 'diningtable', 'pottedplant',
                  'sofa', 'tvmonitor']

object_classes = person_classes + animal_classes
object_classes += vehicle_classes + indoor_classes


class ObjectDetection:
    def __init__(self):
        return

    def set_position(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def similarity_class(self, object_class):
        class_str = object_classes[object_class]
        if(class_str in person_classes):
            return 0
        elif(class_str in animal_classes):
            return 1
        elif(class_str in vehicle_classes):
            return 2
        elif(class_str in indoor_classes):
            return 3

    def set_class(self, object_class):
        self.object_class = object_class
        self.similarity_class = self.similarity_class(object_class)

    def set_ceratinty(self, certainty):
        self.certainty = certainty


def same_class(detection_1, detection_2):
    return detection_1.object_class == detection_2.object_class


def similar_class(detection_1, detection_2):
    return detection_1.similarity_class == detection_2.similarity_class


def iou(detection_1, detection_2):
    boxA = [detection_1.xmin, detection_1.ymin,
            detection_1.xmax, detection_1.ymax]
    boxB = [detection_2.xmin, detection_2.ymin,
            detection_2.xmax, detection_2.ymax]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


prediction_detection_dict = {}
ground_truth_detection_dict = {}


def populate_prediction_detection_dict():
    path = "results/numResults/voc2007Class_"
    for i, object_class in enumerate(object_classes):
        path_object = path + object_class + ".txt"
        detection_lines = []
        with open(path_object, 'r') as f:
            detection_lines = f.readlines()
        for line in detection_lines:
            line_data = line[:-1].split(" ")
            image_num = int(line_data[0])
            certainty = float(line_data[1])
            if(certainty < .25):
                continue
            xmin, ymin, xmax, ymax = tuple(line_data[2:6])
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            new_object_detection = ObjectDetection()
            new_object_detection.set_position(xmin, ymin, xmax, ymax)
            new_object_detection.set_class(i)
            new_object_detection.set_ceratinty(certainty)
            if image_num not in prediction_detection_dict:
                prediction_detection_dict[image_num] = []
            prediction_detection_dict[image_num].append(new_object_detection)
    for image_num in prediction_detection_dict:
        prediction_detection_dict[image_num] = sorted(prediction_detection_dict[image_num],
                                                      key = lambda x: -x.certainty)
    return


def populate_ground_truth_detection_dict():
    path = "results/officialAnnotations"
    for filename in os.listdir(path):
        image_num = int(filename.split(".")[0])
        print image_num
        ground_truth_detection_dict[image_num] = []
        if image_num not in prediction_detection_dict:
            prediction_detection_dict[image_num] = []
        tree = etree.parse(path + "/" + filename)
        root = tree.getroot()
        for child in root:
            if(child.tag != "object"):
                continue
            i = object_classes.index(child.find('name').text)
            xmin, ymin, xmax, ymax = None, None, None, None
            for box in child:
                if(box.tag == "bndbox"):
                    xmin = float(box.find('xmin').text)
                    ymin = float(box.find('ymin').text)
                    xmax = float(box.find('xmax').text)
                    ymax = float(box.find('ymax').text)
            new_object_detection = ObjectDetection()
            new_object_detection.set_position(xmin, ymin, xmax, ymax)
            new_object_detection.set_class(i)
            ground_truth_detection_dict[image_num].append(new_object_detection)
    return


populate_prediction_detection_dict()
populate_ground_truth_detection_dict()


def wrong_class_detection(ground_detection, predicted_detection):
    iou_val = (iou(ground_detection, predicted_detection) > .5)
    is_correct_val = same_class(ground_detection, predicted_detection)
    return iou_val and not is_correct_val


def background_detection(ground_detection, predicted_detection):
    is_correct_val = same_class(ground_detection, predicted_detection)
    return is_correct_val


def similar_class_detection(ground_detection, predicted_detection):
    iou_val = (iou(ground_detection, predicted_detection) > .5)
    is_correct_val = similar_class(ground_detection, predicted_detection)
    return iou_val and is_correct_val


def is_localized_detection(ground_detection, predicted_detection):
    iou_val = (iou(ground_detection, predicted_detection) > .1)
    is_correct_val = same_class(ground_detection, predicted_detection)
    return iou_val and is_correct_val


def is_other_detection(ground_detection, predicted_detection):
    iou_val = (iou(ground_detection, predicted_detection) > .1)
    return iou_val


def is_valid_detection(ground_detection, predicted_detection):
    iou_val = (iou(ground_detection, predicted_detection) > .5)
    is_correct_val = same_class(ground_detection, predicted_detection)
    return iou_val and is_correct_val


def is_correct(remaining_detections, predicted_detection):
    correct_functs = [is_valid_detection]
    is_correct = False
    for ground_detection in remaining_detections:
        for correct_funct in correct_functs:
            is_correct = is_correct or correct_funct(ground_detection,
                                                     predicted_detection)
        if(is_correct):
            remaining_detections.remove(ground_detection)
            break
    return is_correct


def average_precision(ground_detections, predicted_detections):
    if(not len(predicted_detections)):
        return 0.0
    remaining_detections = ground_detections[:]
    for predicted_detection in predicted_detections:
        if(is_correct(remaining_detections, predicted_detection)):
            ground_truth_arr.append(1.0)
        else:
            ground_truth_arr.append(0.0)
        predicted_arr.append(predicted_detection.certainty)
    for remaining_detection in remaining_detections:
        ground_truth_arr.append(1.0)
        predicted_arr.append(0.0)


def is_small_image(detection):
    small_threshold = 1000
    width = detection.xmax - detection.xmin
    length = detection.ymax - detection.ymin
    area = length * width
    return (area < small_threshold)


def is_large_image(detection):
    small_threshold = 1000
    width = detection.xmax - detection.xmin
    length = detection.ymax - detection.ymin
    area = length * width
    return (area > small_threshold)


def in_subset(ground_detections):
    not_in_subset = []
    all_in_subset = True
    for ground_detection in ground_detections:
        for func in not_in_subset:
            if(func(ground_detection)):
                all_in_subset = False
    return all_in_subset

precisions = []
ground_truth_arr = []
predicted_arr = []
for image_num in ground_truth_detection_dict:
    ground_detections = ground_truth_detection_dict[image_num]
    predicted_detections = prediction_detection_dict[image_num]

    if(in_subset(ground_detections)):
        average_precision(ground_detections, predicted_detections)
predicted_arr = [x for _, x in sorted(zip(ground_truth_arr, predicted_arr))]
print average_precision_score(ground_truth_arr, predicted_arr)
