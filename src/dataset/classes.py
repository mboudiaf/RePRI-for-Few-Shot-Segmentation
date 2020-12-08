from collections import defaultdict
import argparse
from typing import Dict, List, Any

classId2className = {'coco': {
                         1: 'person',
                         2: 'bicycle',
                         3: 'car',
                         4: 'motorcycle',
                         5: 'airplane',
                         6: 'bus',
                         7: 'train',
                         8: 'truck',
                         9: 'boat',
                         10: 'traffic light',
                         11: 'fire hydrant',
                         12: 'stop sign',
                         13: 'parking meter',
                         14: 'bench',
                         15: 'bird',
                         16: 'cat',
                         17: 'dog',
                         18: 'horse',
                         19: 'sheep',
                         20: 'cow',
                         21: 'elephant',
                         22: 'bear',
                         23: 'zebra',
                         24: 'giraffe',
                         25: 'backpack',
                         26: 'umbrella',
                         27: 'handbag',
                         28: 'tie',
                         29: 'suitcase',
                         30: 'frisbee',
                         31: 'skis',
                         32: 'snowboard',
                         33: 'sports ball',
                         34: 'kite',
                         35: 'baseball bat',
                         36: 'baseball glove',
                         37: 'skateboard',
                         38: 'surfboard',
                         39: 'tennis racket',
                         40: 'bottle',
                         41: 'wine glass',
                         42: 'cup',
                         43: 'fork',
                         44: 'knife',
                         45: 'spoon',
                         46: 'bowl',
                         47: 'banana',
                         48: 'apple',
                         49: 'sandwich',
                         50: 'orange',
                         51: 'broccoli',
                         52: 'carrot',
                         53: 'hot dog',
                         54: 'pizza',
                         55: 'donut',
                         56: 'cake',
                         57: 'chair',
                         58: 'sofa',
                         59: 'pottedplant',
                         60: 'bed',
                         61: 'diningtable',
                         62: 'toilet',
                         63: 'tv',
                         64: 'laptop',
                         65: 'mouse',
                         66: 'remote',
                         67: 'keyboard',
                         68: 'cell phone',
                         69: 'microwave',
                         70: 'oven',
                         71: 'toaster',
                         72: 'sink',
                         73: 'refrigerator',
                         74: 'book',
                         75: 'clock',
                         76: 'vase',
                         77: 'scissors',
                         78: 'teddy bear',
                         79: 'hair drier',
                         80: 'toothbrush'},

                     'pascal': {
                        1: 'airplane',
                        2: 'bicycle',
                        3: 'bird',
                        4: 'boat',
                        5: 'bottle',
                        6: 'bus',
                        7: 'cat',
                        8: 'car',
                        9: 'chair',
                        10: 'cow',
                        11: 'diningtable',
                        12: 'dog',
                        13: 'horse',
                        14: 'motorcycle',
                        15: 'person',
                        16: 'pottedplant',
                        17: 'sheep',
                        18: 'sofa',
                        19: 'train',
                        20: 'tv'
                        }
                     }

className2classId = defaultdict(dict)
for dataset in classId2className:
    for id in classId2className[dataset]:
        className2classId[dataset][classId2className[dataset][id]] = id


def get_split_classes(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Returns the split of classes for Pascal-5i and Coco-20i
    inputs:
        args

    returns :
         split_classes : Dict.
                         split_classes['coco'][0]['train'] = training classes in fold 0 of Coco-20i
    """
    split_classes = {'coco': defaultdict(dict), 'pascal': defaultdict(dict)}

    # =============== COCO ===================
    name = 'coco'
    class_list = list(range(1, 81))
    split_classes[name][-1]['val'] = class_list
    if args.use_split_coco:
        vals_lists = [list(range(1, 78, 4)), list(range(2, 79, 4)),
                      list(range(3, 80, 4)), list(range(4, 81, 4))]
        for i, val_list in enumerate(vals_lists):
            split_classes[name][i]['val'] = val_list
            split_classes[name][i]['train'] = list(set(class_list) - set(val_list))

    else:
        class_list = list(range(1, 81))
        vals_lists = [list(range(1, 21)), list(range(21, 41)),
                      list(range(41, 61)), list(range(61, 81))]
        for i, val_list in enumerate(vals_lists):
            split_classes[name][i]['val'] = val_list
            split_classes[name][i]['train'] = list(set(class_list) - set(val_list))

    # =============== Pascal ===================
    name = 'pascal'
    class_list = list(range(1, 21))
    vals_lists = [list(range(1, 6)), list(range(6, 11)), list(range(11, 16)), list(range(16, 21))]
    split_classes[name][-1]['val'] = class_list
    for i, val_list in enumerate(vals_lists):
        split_classes[name][i]['val'] = val_list
        split_classes[name][i]['train'] = list(set(class_list) - set(val_list))

    return split_classes


def filter_classes(train_name: str,
                   train_split: int,
                   test_name: str,
                   test_split: int,
                   split_classes: Dict) -> List[int]:
    """ Useful for domain shift experiments. Filters out classes that were seen
        during  training (i.e in the train_name dataset) from the current list.

    inputs:
        train_name : 'coco' or 'pascal'
        test_name : 'coco' or 'pascal'
        train_split : In {0, 1, 2, 3}
        test_split : In {0, 1, 2, 3, -1}. -1 represents "all classes" (the one used in our experiments)
        split_classes: Dict of all classes used for each dataset and each split


    returns :
        kept_classes_id : Filtered list of class ids that will be used for testing
    """
    print(f'INFO: {train_name} -> {test_name}')
    print(f'INFO: {train_split} -> {test_split}')
    print(">> Start Filtering classes ")
    seen_classes = [classId2className[train_name][c] for c in split_classes[train_name][train_split]['train']]
    initial_classes = split_classes[test_name][test_split]['val']
    kept_classes_id = []
    removed_classes = []
    kept_classes_name = []
    for c in initial_classes:
        if classId2className[test_name][c] in seen_classes:
            removed_classes.append(classId2className[test_name][c])
        else:
            kept_classes_id.append(c)
            kept_classes_name.append(classId2className[test_name][c])
    print(">> Removed classes = {} ".format(removed_classes))
    print(">> Kept classes = {} ".format(kept_classes_name))
    return kept_classes_id
