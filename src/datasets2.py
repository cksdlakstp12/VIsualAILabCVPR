import sys,os,json

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

try:
    from . import functional as F
except:
    import functional as F

from utils.utils import *
from utils.transforms import randomHorizontalFlipPropQ

from collections import defaultdict

class KAISTPed(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """
    def __init__(self, args, condition='train'):
        self.args = args
        assert condition in args.dataset.OBJ_LOAD_CONDITIONS
        
        self.mode = condition
        self.image_set = args[condition].img_set
        self.img_transform = args[condition].img_transform
        self.co_transform = args[condition].co_transform        
        self.cond = args.dataset.OBJ_LOAD_CONDITIONS[condition]
        self.annotation = args[condition].annotation
        self._parser = LoadBox()        

        self._annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')

        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  
        
        self.ids = list()
        for line in open(os.path.join('./imageSets', self.image_set)):
            self.ids.append((self.args.path.DB_ROOT, line.strip().split('/')))

    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

        vis, lwir, boxes, labels = self.pull_item(index)
        return vis, lwir, boxes, labels, torch.ones(1,dtype=torch.int)*index, None

    def pull_item(self, index):
        
        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]

        vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ))
        lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
    
        width, height = lwir.size

        # paired annotation
        if self.mode == 'train': 
            vis_boxes = list()
            lwir_boxes = list()

            for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                vis_boxes.append(line.strip().split(' '))
            for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                lwir_boxes.append(line.strip().split(' '))

            vis_boxes = vis_boxes[1:]
            lwir_boxes = lwir_boxes[1:]

            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]

            for i in range(len(vis_boxes)) :
                name = vis_boxes[i][0]
                bndbox = [int(i) for i in vis_boxes[i][1:5]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)) :
                name = lwir_boxes[i][0]
                bndbox = [int(i) for i in lwir_boxes[i][1:5]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        else :
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        ## Apply transforms
        if self.img_transform is not None:
            vis, lwir, boxes_vis , boxes_lwir, _ = self.img_transform(vis, lwir, boxes_vis, boxes_lwir)

        if self.co_transform is not None:
            
            pair = 1

            vis, lwir, boxes_vis, boxes_lwir, pair = self.co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)                      
            if boxes_vis is None:
                boxes = boxes_lwir
            elif boxes_lwir is None:
                boxes = boxes_vis
            else : 
                ## Pair Condition
                ## RGB / Thermal
                ##  1  /  0  = 1
                ##  0  /  1  = 2
                ##  1  /  1  = 3
                if pair == 1 :
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 3
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 3
                else : 
                    
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 1
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 2
                
                boxes = torch.cat((boxes_vis,boxes_lwir), dim=0)
                boxes = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes.numpy()]))))   

        ## Set ignore flags
        ignore = torch.zeros( boxes.size(0), dtype=torch.bool)
               
        for ii, box in enumerate(boxes):
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore[ii] = 1
        
        boxes[ignore, 4] = -1
        
        labels = boxes[:,4]
        boxes_t = boxes[:,0:4]
        #print("labels : ", labels)
        return vis, lwir, boxes_t, labels

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        vis = list()
        lwir = list()
        boxes = list()
        labels = list()
        index = list()

        for b in batch:
            vis.append(b[0])
            lwir.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            index.append(b[4])

        vis = torch.stack(vis, dim=0)
        lwir = torch.stack(lwir, dim=0)
  
        return vis, lwir, boxes, labels, index  

class KAISTPedWSEpoch(KAISTPed):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """
    def __init__(self, args, aug_mode, sample = None, condition='train'):
        super().__init__(args, condition)
        self.pair = 1
        self.aug_mode = aug_mode
        self.soft_update_mode = args[condition].soft_update_mode
        self.teacher_image_set = args[condition].teacher_img_set
        if aug_mode == "weak":
            self.image_set = self.teacher_image_set
        self.weak_transform = args[condition].weak_transform 
        # self.weak4strong_transform = args[condition].weak4strong_transform 
        self.strong_transform = args[condition].strong_transform 
        self.annotations = defaultdict(list)
        self.min_score = args[condition].min_score
        self.props_path_format = os.path.join(args.jobs_dir, 'props_%d.txt')
        self.props_path = None

        
        with open(os.path.join('./imageSets', self.teacher_image_set), 'r') as f:
            self.teacher_image_ids = {idx+1: line.strip() for idx, line in enumerate(f)}

        if sample == "Labeled":
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.L_image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
        elif sample == "Unlabeled":
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.U_image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
        else:
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
            
        self._annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')
        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')

    def set_propFilePath_by_epoch(self, epoch):
        self.props_path = self.props_path_format % epoch

    def parse_teacher_inference(self, results):
        """
        This method for strong augmentation that using student model.
        After teacher model inference non labeled data, 
            parsing that result to be compatible with student model's input type
        Parsing result save into self.annotations variable, 
            it will called to determine the augmentation type(weak, strong).
        This method should be used after teacher model inference.
        """
        self.annotations = defaultdict(list)
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                img_id = self.teacher_image_ids[image_id]
                if score >= self.min_score:
                    self.annotations[img_id].append([float(x), float(y), float(w), float(h), float(score)])
    
    def load_propFile(self):
        # Load flip probs
        self.props = dict()
        assert self.props_path is not None, "please call the set_propfile_path_by_epoch method before start epoch"
        with open(self.props_path, "r") as f:
            for line in f.readlines():
                index, prop = line.strip().split(",")
                self.props[index] = prop
        os.remove(self.props_path)

    def __getitem__(self, index):
        vis, lwir, vis_box, lwir_box, vis_labels,lwir_labels, is_annotation = self.pull_item(index)
        ##print(self.ids[index])
        ##print(vis, lwir, boxes, labels, torch.ones(1,dtype=torch.int)*index, self.ids[index])
        return vis, lwir, vis_box, lwir_box, vis_labels, lwir_labels, torch.ones(1,dtype=torch.int)*index, is_annotation

    def pull_item(self, index):
        global randomHorizontalFlipPropQ
        is_annotation = True

        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]

        if self.aug_mode == "weak":
            flipProp = random.random()
            randomHorizontalFlipPropQ.append(flipProp)
            with open(self.props_path, "a") as f:
                f.write(f"{index},{flipProp}\n")
        else:
            if index in self.props:
                randomHorizontalFlipPropQ.append(self.props[index])
            else:
                randomHorizontalFlipPropQ.append(random.random())

        vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ))
        lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
    
        width, height = lwir.size

        if self.mode == 'train': 
            vis_boxes = list()
            lwir_boxes = list()
            id = f"{set_id}/{vid_id}/{img_id}"
            if id in self.teacher_image_ids.values() and self.aug_mode == "strong":
                # Load bounding boxes from pre-inferred results
                boxes = self.annotations[id][0:4]

                vis_boxes = np.array(boxes, dtype=np.float)
                lwir_boxes = np.array(boxes, dtype=np.float)

                is_annotation = False

            else:
                for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                    vis_boxes.append(line.strip().split(' '))
                for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                    lwir_boxes.append(line.strip().split(' '))

                vis_boxes = vis_boxes[1:]
                lwir_boxes = lwir_boxes[1:]

            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            
            compatibilityIndex = 0
            if is_annotation:
                compatibilityIndex = 1
                
            for i in range(len(vis_boxes)):
                bndbox = [int(i) for i in vis_boxes[i][0+compatibilityIndex:4+compatibilityIndex]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)) :
                ##print(f"lwir : {lwir_boxes}\n")
                name = lwir_boxes[i][0]
                bndbox = [int(i) for i in lwir_boxes[i][0+compatibilityIndex:4+compatibilityIndex]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

            #print(f"boxes_vis : {boxes_vis}")
            #print(f"boxes_lwir : {boxes_lwir}")

        else :
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        ## Apply transforms
        if self.aug_mode == "weak":
            vis, lwir, boxes_vis, boxes_lwir, _ = self.weak_transform(vis, lwir, boxes_vis, boxes_lwir, self.pair)
        else:
            # vis, lwir, boxes_vis, boxes_lwir, _ = self.weak4strong_transform(vis, lwir, boxes_vis, boxes_lwir, self.pair)
            vis, lwir, boxes_vis, boxes_lwir, _ = self.strong_transform(vis, lwir, boxes_vis, boxes_lwir, self.pair)

        if self.co_transform is not None:

            # vis, lwir, boxes_vis, boxes_lwir, pair = self.co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)                      
            if boxes_vis is None:
                #print("vis none")
                boxes = boxes_lwir
            elif boxes_lwir is None:
                #print("lwir none")
                boxes = boxes_vis
            else : 
                ## Pair Condition
                ## RGB / Thermal
                ##  1  /  0  = 1
                ##  0  /  1  = 2
                ##  1  /  1  = 3
                if self.pair == 1 :
                    #print("Before vis_box : ", boxes_vis)
                    #print("Befor lwir_box : ", boxes_lwir)
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 3
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 3
                    #print("after vis_box : ", boxes_vis)
                    #print("after lwir_box : ", boxes_lwir)
                else :
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 1
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 2
                #print("boxes_vis :", boxes_vis)
                #print("boxes_lwir : ", boxes_lwir)
                #boxes = torch.cat((boxes_vis,boxes_lwir), dim=0)
                ##print("before : ",boxes)
                #boxes = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes.numpy()])))) 
                ##print("after : ",boxes)

        ## Set ignore flags
        ignore_vis = torch.zeros(boxes_vis.size(0), dtype=torch.bool)
        ignore_lwir = torch.zeros(boxes_lwir.size(0), dtype=torch.bool)
        
        for ii, box in enumerate(boxes_vis):
            ##print("boxes :",boxes)
            #print("vis_box : ", box)
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore_vis[ii] = 1

        for ii, box in enumerate(boxes_lwir):
            ##print("boxes :",boxes)
            #print("lwir_box : ", box)
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore_lwir[ii] = 1
        
        boxes_vis[ignore_vis, 4] = -1
        boxes_lwir[ignore_lwir, 4] = -1

        ##print(f"여기 박스 : {boxes}")
        vis_labels = boxes_vis[:,4]
        lwir_labels = boxes_lwir[:,4]
        ##print(f"여기 레이블 : {labels}")
        boxes_vis = boxes_vis[:,0:4]
        boxes_lwir = boxes_lwir[:,0:4]
        #print("vis_labels : ", vis_labels)
        #print("lwir_labels : ", lwir_labels)
        ##print(f"boxes_t : {boxes_t}")
        return vis, lwir, boxes_vis, boxes_lwir, vis_labels, lwir_labels, is_annotation
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        vis = list()
        lwir = list()
        vis_box = list()
        lwir_box = list()
        vis_labels = list()
        lwir_labels = list()
        index = list()
        is_anno = list()

        for b in batch:
            vis.append(b[0])
            lwir.append(b[1])
            vis_box.append(b[2])
            lwir_box.append(b[3])
            vis_labels.append(b[4])
            lwir_labels.append(b[5])
            index.append(b[6])
            is_anno.append(b[7])

        vis = torch.stack(vis, dim=0)
        lwir = torch.stack(lwir, dim=0)
  
        return vis, lwir, vis_box, lwir_box, vis_labels, lwir_labels, index, is_anno

class KAISTPedWSIter(KAISTPedWSEpoch):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """
    def __init__(self, args, condition='train', sample = None):
        super().__init__(args, "strong", condition)
        self.weak_transform = args[condition].iter_weak_transform 
        self.strong_transform = args[condition].iter_strong_transform 
        self.baseline_transform = args[condition].baseline_transform
        self.L_image_set = args[condition].L_img_set
        self.U_image_set = args[condition].U_img_set

        if sample == "Labeled":
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.L_image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
        elif sample == "Unlabeled":
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.U_image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
        else:
            self.ids = list()
            for line in open(os.path.join('./imageSets', self.image_set)):
                self.ids.append(('../data/kaist-rgbt/', line.strip().split('/')))
            
        self._annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')
        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')
    def __getitem__(self, index):
        w_vis, w_lwir, s_vis, s_lwir, boxes, labels, is_annotation = self.pull_item(index)
        ##print(self.ids[index])
        ##print(vis, lwir, boxes, labels, torch.ones(1,dtype=torch.int)*index, self.ids[index])
        return w_vis, w_lwir, s_vis, s_lwir, boxes, labels, torch.ones(1,dtype=torch.int)*index, is_annotation
    
    def pull_item(self, index):
        is_annotation = True

        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]

        vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ))
        lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
    
        width, height = lwir.size

        if self.mode == 'train': 
            vis_boxes = list()
            lwir_boxes = list()
            id = f"{set_id}/{vid_id}/{img_id}"
            
            #print()
            #print(len(self.ids))
            #print(index)
            #print(id in self.teacher_image_ids.values())
            #print(id)
            #print()
            
            prop = random.random()
            if id in self.teacher_image_ids.values():
                # Load bounding boxes from pre-inferred results
                is_annotation = False
                self.weak_transform.update_prop(prop)
                self.strong_transform.update_prop(prop)
                w_vis, w_lwir, _, _, _ = self.weak_transform(vis, lwir, None, None, self.pair)
                s_vis, s_lwir, _, _, _ = self.strong_transform(vis, lwir, None, None, self.pair)
                #print(vis_boxes, lwir_boxes)
                return w_vis, w_lwir, s_vis, s_lwir, None, None, is_annotation
            else:
                for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                    vis_boxes.append(line.strip().split(' '))
                for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                    lwir_boxes.append(line.strip().split(' '))

                #print(f"before vis : {vis_boxes}")
                #print(f"before lwir : {lwir_boxes}")

                vis_boxes = vis_boxes[1:]
                lwir_boxes = lwir_boxes[1:]

                #print(f"after vis : {vis_boxes}")
                #print(f"after lwir : {lwir_boxes}")
                
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]

            compatibilityIndex = 0
            if is_annotation:
                compatibilityIndex = 1
                
            for i in range(len(vis_boxes)):
                bndbox = [int(i) for i in vis_boxes[i][0+compatibilityIndex:4+compatibilityIndex]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)) :
                ##print(f"lwir : {lwir_boxes}\n")
                name = lwir_boxes[i][0]
                bndbox = [int(i) for i in lwir_boxes[i][0+compatibilityIndex:4+compatibilityIndex]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

            #print(f"boxes_vis : {boxes_vis}")
            #print(f"boxes_lwir : {boxes_lwir}")

        else :
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        self.strong_transform.update_prop(prop)
        ## Apply transforms
        vis, lwir, boxes_vis, boxes_lwir, _ = self.baseline_transform(vis, lwir, boxes_vis, boxes_lwir, self.pair)
        if self.co_transform is not None:

            # vis, lwir, boxes_vis, boxes_lwir, pair = self.co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)                      
            if boxes_vis is None:
                #print("vis none")
                boxes = boxes_lwir
            elif boxes_lwir is None:
                #print("lwir none")
                boxes = boxes_vis
            else : 
                ## Pair Condition
                ## RGB / Thermal
                ##  1  /  0  = 1
                ##  0  /  1  = 2
                ##  1  /  1  = 3
                if self.pair == 1 :
                    """
                    if is_annotation == False:
                        print("Before sup vis_box : ", boxes_vis)
                        print("Befor sup lwir_box : ", boxes_lwir)
                    else:
                        print("Before un vis_box : ", boxes_vis)
                        print("Befor un lwir_box : ", boxes_lwir)
                    """
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 3
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 3
                    #if is_annotation == True:
                        #print("boxes_vis bef : ", boxes_vis)
                        #print("boxes_lwir bef : ", boxes_lwir)
                    #if is_annotation == False:
                     #   print("after vis_box : ", boxes_vis)
                     #   print("after lwir_box : ", boxes_lwir)
                else :
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 1
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 2
                    
                #print("boxes_vis :", boxes_vis)
                #print("boxes_lwir : ", boxes_lwir)
                boxes = torch.cat((boxes_vis,boxes_lwir), dim=0)
                ##print("before : ",boxes)
                boxes = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes.numpy()])))) 
                ##print("after : ",boxes)

        ## Set ignore flags
        ignore = torch.zeros( boxes.size(0), dtype=torch.bool)
               
        for ii, box in enumerate(boxes):
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore[ii] = 1
        
        boxes[ignore, 4] = -1
        
        labels = boxes[:,4]
        boxes_t = boxes[:,0:4]
        #if is_annotation == False:
        #print("isanno / vis_labels : ", is_annotation ,vis_labels)
        #print("isanno / lwir_labels : ", is_annotation, lwir_labels)
        ##print(f"boxes_t : {boxes_t}")

        return vis, lwir, vis, lwir, boxes_t, labels, is_annotation
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        w_vis = list()
        w_lwir = list()
        s_vis = list()
        s_lwir = list()
        boxes = list()
        labels = list()
        index = list()
        is_anno = list()

        for b in batch:
            w_vis.append(b[0])
            w_lwir.append(b[1])
            s_vis.append(b[2])
            s_lwir.append(b[3])
            boxes.append(b[4])
            labels.append(b[5])
            index.append(b[6])
            is_anno.append(b[7])

        w_vis = torch.stack(w_vis, dim=0)
        w_lwir = torch.stack(w_lwir, dim=0)
        s_vis = torch.stack(s_vis, dim=0)
        s_lwir = torch.stack(s_lwir, dim=0)
  
        return w_vis, w_lwir, s_vis, s_lwir, boxes, labels, index, is_anno
class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='xyxy'):
        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                
        res = [ [0, 0, 0, 0, -1] ]

        for obj in target.iter('object'):           
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]

            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )

            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            
            bndbox.append(1)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]
            
        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


if __name__ == '__main__':
    """Debug KAISTPed class"""
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from utils.functional import to_pil_image, unnormalize
    import config

    def draw_boxes(axes, boxes, labels, target_label, color):
        for x1, y1, x2, y2 in boxes[labels == target_label]:
            w, h = x2 - x1 + 1, y2 - y1 + 1
            axes[0].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))

    args = config.args
    test = config.test

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    dataset = KAISTPed(args, condition='test')

    # HACK(sohwang): KAISTPed always returns empty boxes in test mode
    dataset.mode = 'train'

    vis, lwir, boxes, labels, indices = dataset[1300]

    vis_mean = dataset.co_transform.transforms[-2].mean
    vis_std = dataset.co_transform.transforms[-2].std

    lwir_mean = dataset.co_transform.transforms[-1].mean
    lwir_std = dataset.co_transform.transforms[-1].std

    # C x H x W -> H X W x C
    vis_np = np.array(to_pil_image(unnormalize(vis, vis_mean, vis_std)))
    lwir_np = np.array(to_pil_image(unnormalize(lwir, lwir_mean, lwir_std)))

    # Draw images
    axes[0].imshow(vis_np)
    axes[1].imshow(lwir_np)
    axes[0].axis('off')
    axes[1].axis('off')

    # Draw boxes on images
    input_h, input_w = test.input_size
    xyxy_scaler_np = np.array([[input_w, input_h, input_w, input_h]], dtype=np.float32)
    boxes = boxes * xyxy_scaler_np

    draw_boxes(axes, boxes, labels, 3, 'blue')
    draw_boxes(axes, boxes, labels, 1, 'red')
    draw_boxes(axes, boxes, labels, 2, 'green')

    frame_id = dataset.ids[indices.item()]
    set_id, vid_id, img_id = frame_id[-1]
    fig.savefig(f'{set_id}_{vid_id}_{img_id}.png')
