from PIL import Image
import numpy as np
import torch
import os
import  xml.dom.minidom
label_mapping={'background': 0, 'person': 15, 'motorbike': 14, 'train': 19, 'cow': 10, 'boat': 4, 'diningtable': 11, 'sofa': 18, 'cat': 8, 'horse': 13, 'dog': 12, 'tvmonitor': 20, 'bottle': 5, 'pottedplant': 16, 'bird': 3, 'car': 7, 'bus': 6, 'bicycle': 2, 'chair': 9, 'sheep': 17, 'aeroplane': 1}

class VOCDataset(object):
    def __init__(self,root,transforms=None):
        self.root=root
        self.transforms=transforms
        self.xmls=list(sorted(os.listdir(os.path.join(root,"Annotations"))))
        self.imgs=list(sorted(os.listdir(os.path.join(root,"JPEGImages"))))
        self.masks=list(sorted(os.listdir(os.path.join(root,"SegmentationObject"))))
    def __getitem__(self,idx):
        xmlpath=os.path.join(self.root,"Annotations",self.xmls[idx])
        imagepath=os.path.join(self.root,"JPEGImages",self.imgs[idx])
        maskpath=os.path.join(self.root,"SegmentationObject",self.masks[idx])
        dom=xml.dom.minidom.parse(xmlpath)
        root=dom.documentElement
        obs=root.getElementsByTagName('object')
        labels=[]
        for ob in obs:
            label=ob.getElementsByTagName("name")[0].firstChild.data
            labels.append(label_mapping[label])
        img=Image.open(imagepath).convert("RGB")
        mask=Image.open(maskpath)
        mask=np.array(mask)
        obj_ids=np.unique(mask)
        obj_ids=obj_ids[1:-1]
        masks=mask==obj_ids[:,None,None]
        num_objs=len(obj_ids)
        boxes=[]
        for i in range(num_objs):
            pos=np.where(masks[i])
            xmin=np.min(pos[1])
            xmax=np.max(pos[1])
            ymin=np.min(pos[0])
            ymax=np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.tensor(labels)
        masks=torch.as_tensor(masks,dtype=torch.uint8)
        image_id=torch.tensor([idx])
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd=torch.zeros((num_objs,),dtype=torch.int64)
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["masks"]=masks
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd
        if self.transforms is not None:
            img,target=self.transforms(img, target)
        return img,target
    def __len__(self):
        return len(self.imgs)
"""
mask=np.array(mask)
obj_ids=np.unique(mask)
obj_ids=obj_ids[1:]
masks=mask == obj_ids[:, None, None]
num_objs = len(obj_ids)
boxes = []
for i in range(num_objs):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    boxes.append([xmin, ymin, xmax, ymax])
boxes = torch.as_tensor(boxes, dtype=torch.float32)
labels = torch.ones((num_objs,), dtype=torch.int64)
masks = torch.as_tensor(masks, dtype=torch.uint8)
image_id = torch.tensor([idx])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
target = {}
target["boxes"] = boxes
target["labels"] = labels
target["masks"] = masks
target["image_id"] = image_id
target["area"] = area
target["iscrowd"] = iscrowd
"""

# num_objs为图片中目标数量
# boxes形状为[num_objs,4]，对应每个目标的框
# labels形状为[num_objs]为每个目标的分类
# masks形状为[num_objs,H,W]为每个目标的0-1分割
# area形状为[num_objs]为每个框的尺寸
# iscrowd形状为[num_objs]为每个目标是否重叠