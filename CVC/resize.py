from PIL import Image
import glob

def resize_and_save_image(image_path, desired_size):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 이미지 크기 변경
    img_resized = img.resize(desired_size)
    img_resized.save(image_path)

def adjust_annotation(annotation_path, resize_ratio):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    adjusted_lines = []
    for line in lines:
        anno = list(map(int, line.split()))
        
        # 주석 조정
        anno[1] = int(anno[1] * resize_ratio)
        anno[3] = int(anno[3] * resize_ratio)
        
        adjusted_lines.append(" ".join(map(str, anno)) + "\n")

    with open(annotation_path, 'w') as file:
        file.writelines(adjusted_lines)

desired_size = (640, 512)

original_height = 471
new_height = 512
resize_ratio = new_height / original_height

path = "C:/Users/USER/Desktop/학부연구생/data/CVC-14/**/FramesPos/*.tif"
filepaths = glob.glob(path, recursive=True)
for filepath in filepaths:
    image_path = filepath
    annotation_path = filepath.replace("FramesPos", "Annotations").replace("tif", "txt")
    resize_and_save_image(image_path, desired_size)
    adjust_annotation(annotation_path, resize_ratio) 
