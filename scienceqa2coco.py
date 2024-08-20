import json
import glob, os
import cv2
from scipy.spatial import ConvexHull
import numpy as np
from tqdm import tqdm
# import jsonlines
import pdb

stone_dict = {'info':{"description": "Scienceqa Dataset",
                      "url": "None",
                      "version": "1.0",
                      "year": 2023,
                      "contributor": "None",
                      "date_created": "2023/09/23"},
              # 'images':[{"license": 3,
              #            "file_name": "stone.jpg",
              #            "coco_url": "/data/luohh/cv/stone_image/stone.jpg",
              #            "height": 3648,
              #            "width": 2736,
              #            "date_captured": "2023-9-13 15:09:45",
              #            "flickr_url": "/data/luohh/cv/stone_image/stone.jpg",
              #            "id": 1},],
              'licenses':[{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'},
                          {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'},
                          {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'},
                          {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'},
                          {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'},
                          {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'},
                          {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'},
                          {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}],
              # 'annotations':[{'segmentation': [[296.65, 388.33, 296.65, 388.33, 297.68, 388.33, 297.68, 388.33]], 'area': 0.0, 'iscrowd': 0, 'image_id': 200365, 'bbox': [296.65, 388.33, 1.03, 0.0], 'category_id': 58, 'id': 918}],
              'categories':[{'supercategory': 'outdoor', 'id': 1, 'name': 'image'}]
              }

def get_bbox(ps):
    ys = ps[0:len(ps):2]
    xs = ps[1:len(ps):2]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    return [y_min, x_min, y_max, x_max]
def get_area(polygon):
    polygon = np.array(polygon).astype("float32")
    area = 0
    q = polygon[-1]
    for p in polygon:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area) / 2.0
# print(get_area([[0,0],[0,2],[2,0]]))
# print(get_bbox([1,2,3,4,5,6]))
# pdb.set_trace()
# examples = json.load(open('/data/luohh/cv/annotations/instances_val2014.json'))
# for example in examples['annotations']:
#     print(example)
#     pdb.set_trace()
# stone_files = glob.glob("/data/luohh/cv/stone_annotation/*.json")
data_root = '/data/luohh/acl23/data/scienceqa/'
problems = json.load(open(os.path.join(data_root, 'problems_blip2xl_angle.json')))
split = 'val'
stone_annotations = []
stone_images = []
stone_count = 0
for qid in tqdm(problems):
    if problems[qid]['image'] != None and problems[qid]['split'] == split:
        image_file = os.path.join(data_root, problems[qid]['split'], str(qid), problems[qid]['image'])
        image = cv2.imread(image_file)
        height, width = image.shape[0], image.shape[1]
        stone_images.append({"license": 3,
                             "file_name": image_file.split('/')[-1],
                             "coco_url": image_file,
                             "height": height,
                             "width": width,
                             "date_captured": "2023-9-13 15:09:45",
                             "flickr_url": image_file,
                             "id": int(qid)})
        v = [0,0,0,width-1,height-1,0,height-1,width-1,0,0]
        area_v = []
        new_v = []
        for i in range(0, len(v) - 2, 2):
            area_v.append([v[i], v[i + 1]])
            new_v.extend([height - v[i + 1], v[i]])
        area = get_area(area_v)
        stone_annotations.append({'segmentation': [new_v],
                                  'area': area,
                                  'iscrowd': 0,
                                  'image_id': int(qid),
                                  'bbox': get_bbox(new_v),
                                  'category_id': 1,
                                  'id': stone_count})
        stone_count += 1

stone_dict.update({"images":stone_images,"annotations":stone_annotations})
filename = f'/data/luohh/acl23/coco/annotations/scienceqa_{split}2023.json'
with open(filename,'w') as f:
    json.dump(stone_dict, f)
print('Done!')
# 'info', 'images', 'licenses', 'annotations', 'categories'
# print(len(examples['annotations']))
# print(len(examples['images']))
# img_id = []

# print(examples['annotations'][0])


