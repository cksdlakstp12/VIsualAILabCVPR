import glob
import os
import json

results = {}
info = {
    "dataset": "CVC-14: Visible-FIR Day-Night Pedestrian Sequence Dataset",
    "url": "http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/",
    "related_project_url": "",
    "publish": ""
}
info_improved = {
    "sanitized_annotation":{
        "publish": "",
        "url":"",
        "target":""
    },
    "improved_annotation":{
        "url":"",
        "publish":"",
        "target":""
    }
}

categories = [
    {
        "id": 0,
        "name": "__ignore__"
    },
    {
        "id": 1,
        "name": "person"
    },
    {
        "id": 2,
        "name": "cyclist"
    },
    {
        "id": 3,
        "name": "people"
    },
    {
        "id": 4,
        "name": "person?"
    }
]

results["info"] = info
results["info_improved"] = info_improved
results["categories"] = categories

images = []
annotations = []

def make_path(filepath):
    split = filepath.strip().split("/")
    path = os.path.join('../data/CVC-14', '%s', '%s', 'NewTest', 'Annotations', '%s.txt')
    return path % (split[0], "Visible", split[2])

txt_path = "src/imageSets/CVC-test.txt"
with open(txt_path, "r") as f:
    filepaths = f.readlines()

anno_idx = 0
for idx, filepath in enumerate(filepaths):

    images.append({
        "id":idx,
        "im_name":filepath.strip(),
        "height":512,
        "width":640
    })

    with open(make_path(filepath), "r") as f:
        for line in f.readlines():
            map2int = list(map(int, line.strip().split(" ")))
            annotations.append({
                "id":anno_idx,
                "image_id":idx,
                "category_id":1,
                "bbox":map2int[:4],
                "height":map2int[3],
                "occlusion":0,
                "ignore":0 if map2int[4] == 1 else 1
            })
            anno_idx += 1

results["images"] = images
results["annotations"] = annotations

json_object = json.dumps(results, indent=4)
with open("CVC14_annotations_test.json", "w") as outfile:
    outfile.write(json_object)
