import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from transformers import ViTModel, ViTConfig
from PIL import Image
import os
from collections import defaultdict
import torch
from torch import nn
from torchvision.transforms import Compose
import matplotlib.patches as patches
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision.transforms import ToPILImage
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
warnings.filterwarnings('ignore')

model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Finetuning the ViT Layers and the embeddings for optimal performance
for name, param in vit_backbone.named_parameters():
    if 'embeddings' in name or 'encoder.layer' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

class ObjectDetectionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, num_boxes, hidden_layer_size=1024):
        super(ObjectDetectionHead, self).__init__()

        # Single fully connected layer with BatchNorm
        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size)  # BatchNorm layer
        )

        # Final layer for class predictions
        self.classifier = nn.Linear(hidden_layer_size, num_classes * num_boxes)

        # Final layer for bounding box regression
        self.box_regressor = nn.Linear(hidden_layer_size, 4 * num_boxes)

        # Activation function for bounding box coordinates
        self.box_regressor_activation = nn.Sigmoid()

    def forward(self, x):
        # Pass input through the fully connected layer
        x = self.fc_layer(x)

        # Class logits and bounding box coordinates
        class_logits = self.classifier(x)
        box_coordinates = self.box_regressor_activation(self.box_regressor(x))

        return class_logits, box_coordinates

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, num_boxes):
        super(ObjectDetectionModel, self).__init__()
        # Load the ViT model as a backbone
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)

        # The size of the embeddings generated by the backbone
        embedding_size = 768

        # Object detection head with separate BatchNorm for class_logits and box_coordinates
        self.head = ObjectDetectionHead(embedding_size, num_classes, num_boxes)

        # BatchNorm layer for class_logits
        self.class_bn = nn.BatchNorm1d(num_classes * num_boxes)

    def forward(self, pixel_values):
        # Forward pass through the backbone
        outputs = self.backbone(pixel_values=pixel_values)

        # We use the representation of the [CLS] token (first token) for detection
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Forward pass through the head
        class_logits, box_coordinates = self.head(cls_token)

        # Apply BatchNorm separately to class_logits
        class_logits = self.class_bn(class_logits)

        return class_logits, box_coordinates

class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        # Check if the 'categories' key exists and print its contents
       # if 'categories' in coco:
       #     print("Found 'categories' in annotations:", coco['categories'])
       # else:
       #     print("'categories' not found in annotations.")    
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        if 'categories' in coco:
            for cat in coco['categories']:
                self.cat_dict[cat['id']] = cat
        
        # Check if cat_dict is populated
      #  print("cat_dict contents:", self.cat_dict)
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        self.create_id_to_name_mapping()
        self.create_category_mapping()

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

    def create_category_mapping(self):
        """
        Create a mapping from COCO category IDs to sequential indices.
        """
        all_cat_ids = sorted([cat_id for cat_id in self.cat_dict.keys()])
        self.cat_id_to_index = {cat_id: index for index, cat_id in enumerate(all_cat_ids)}
       # print("Category IDs in mapping:", sorted(self.cat_id_to_index.keys())) # debugging purposes

    def create_id_to_name_mapping(self):
        """
        Create a mapping from COCO category IDs to category names.
        """
        self.id_to_name = {cat['id']: cat['name'] for cat in self.cat_dict.values()}

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    
    def get_index_from_name(self, name):
        for cat_id, cat_name in self.id_to_name.items():
            if cat_name == name:
                return self.cat_id_to_index[cat_id]
        return -1  # Return an invalid index for unknown category names

    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        

    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
    
    def load_bbox_and_labels(self, img_id):
        """
        Load bounding boxes and labels for a given image ID.
        """
        annotations = self.annIm_dict[img_id]
        bboxes = []
        labels = []

        for ann in annotations:
            # COCO bbox format: [x_min, y_min, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            bboxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])

        return bboxes, labels

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.coco_parser = COCOParser(annotation_file, image_dir)
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.coco_parser.get_imgIds())

    def convert_and_normalize_boxes(self, bboxes, img_width, img_height):
        normalized_bboxes = []
        for x, y, w, h in bboxes:
            x_min = max(0, x) / img_width
            y_min = max(0, y) / img_height
            x_max = min(img_width, x + w) / img_width
            y_max = min(img_height, y + h) / img_height
            normalized_bboxes.append([x_min, y_min, x_max, y_max])
        return normalized_bboxes
    
    def __getitem__(self, idx):
        # Get image ID and path
        img_id = self.coco_parser.get_imgIds()[idx]
        img_path = os.path.join(self.image_dir, self.coco_parser.im_dict[img_id]['file_name'])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # Original image size (W, H)
        
        # Load bounding boxes and labels
        bboxes, labels = self.coco_parser.load_bbox_and_labels(img_id)
        
        # Normalize and convert bounding boxes
        bboxes = self.convert_and_normalize_boxes(bboxes, *original_size)

        # Convert category IDs to names
        label_names = [self.coco_parser.id_to_name[label] for label in labels]
        
        # Map COCO category IDs to sequential indices
        labels = [self.coco_parser.cat_id_to_index[label] for label in labels]
        
        # Convert bboxes and labels to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"image_id": img_id, "boxes": bboxes, "cat_id": labels, "labels": label_names, "image_size": original_size}
        
        # Apply transformations to the image
        if self.transform:
            transformed_image = self.transform(image)
        
        # Return the transformed image and the target
        return transformed_image, target


transform = Compose([
    Resize((224, 224)),  # Resize images
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    images = [item[0] for item in batch]  # Extract images
    targets = [item[1] for item in batch]  # Extract targets (dicts of 'boxes' and 'labels')

    images = torch.stack(images, dim=0)

    # since targets can have different numbers of objects,
    # we don't stack or pad them like images.
    return images, targets

def map_indices_to_labels(label_indices, label_mapping):
    """
    Map a tensor of label indices to actual labels using the provided mapping.

    :param label_indices: Tensor of label indices.
    :param label_mapping: Dict mapping indices to labels.
    :return: List of labels.
    """
    return [label_mapping[idx.item()] for idx in label_indices]

def visualize_bboxes(image, target):
    # Extract the bounding boxes and labels from the target dictionary
    boxes = target["boxes"]  # Assuming this is a tensor
    labels = target["labels"]
    original_size = target["image_size"]  # Assuming original size is stored in target

    # Convert the image to a NumPy array and permute its dimensions if necessary
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        img_height, img_width = image_np.shape[:2]
    else:  # if it's a PIL Image or a NumPy array already
        image_np = np.array(image)
        img_width, img_height = image_np.shape[:2]

    # Create a figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    # Loop through each bounding box and label
    for box, label in zip(boxes, labels):
        # Make sure the box tensor is on the CPU and convert it to a NumPy array
        box = box.cpu().numpy()

        # Denormalize the coordinates
        x_min, y_min, x_max, y_max = box
        x_min *= img_width
        y_min *= img_height
        x_max *= img_width
        y_max *= img_height

        # Calculate the width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Add label text
        ax.text(x_min, y_min, label, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Show the image with bounding boxes
    plt.show()
    # Save the image with bounding boxes
    plt.savefig("/mnt/offload/mansi/VitObjDet/visualize_ground_truth.png")

def visualize_predictions(image_tensor, boxes, labels, scores, image_size, threshold = 0.5):
    # Convert image tensor to PIL Image
    image = ToPILImage()(image_tensor)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Clip the boxes to the image size and rescale
    boxes = np.clip(boxes, 0, 1)  # Ensuring the boxes are within [0, 1] range
    boxes[:, [0, 2]] *= image_size[1]  # Rescale x coordinates to image width
    boxes[:, [1, 3]] *= image_size[0]  # Rescale y coordinates to image height

    for box, label, score in zip(boxes, labels, scores):
        # Check if the box has a valid shape
        if box[2] > box[0] and box[3] > box[1] and score >= threshold:
            # Draw the box
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            # Position the text label
            label_x = box[0]
            label_y = box[1] - 10 if box[1] - 10 > 0 else box[1] + 20
            ax.text(label_x, label_y, f"{label}: {score:.2f}", fontsize=12, color='white', 
                    bbox=dict(facecolor='yellow', alpha=0.75))
    
    plt.axis('off')  # Hide the axes
    plt.show()
    plt.savefig("/mnt/offload/mansi/VitObjDet/visualize_predictions.png")


def compute_loss(class_logits, box_coordinates, targets, coco_parser, alpha=0.25, num_classes=81, num_boxes=100):
    classification_loss = nn.CrossEntropyLoss(reduction='sum')
    regression_loss = nn.SmoothL1Loss(reduction='sum')

    batch_size = class_logits.size(0)
    total_class_loss = 0
    total_reg_loss = 0
    valid_images = 0

    class_logits = class_logits.view(batch_size, num_boxes, num_classes)
    box_coordinates = box_coordinates.view(batch_size, num_boxes, 4)

    for i in range(batch_size):
        # Convert boxes to tensor if they're not already
        target_boxes = torch.as_tensor(targets[i]['boxes'], dtype=torch.float32, device=box_coordinates.device)
        num_objs = target_boxes.shape[0]
        if num_objs == 0:
            continue

        current_logits = class_logits[i, :num_objs]
        current_boxes = box_coordinates[i, :num_objs]

        current_logits = current_logits.view(-1, num_classes)
        current_boxes = current_boxes.view(-1, 4)

        # Ensure labels are tensors
        labels_tensor = targets[i]['labels']
        if isinstance(labels_tensor, list):
            labels_indices = [coco_parser.get_index_from_name(label) for label in targets[i]['labels']]
            labels_tensor = torch.tensor(labels_indices, dtype=torch.int64, device=current_logits.device)
        if current_logits.size(0) == 0:
            continue

        class_loss = classification_loss(current_logits, labels_tensor)
        reg_loss = regression_loss(current_boxes, target_boxes)

        total_class_loss += class_loss
        total_reg_loss += reg_loss
        valid_images += 1

    if valid_images == 0:
        return torch.tensor(0.0, device=box_coordinates.device)

    avg_class_loss = total_class_loss / valid_images
    avg_reg_loss = total_reg_loss / valid_images

    total_loss = alpha * avg_class_loss + (1 - alpha) * avg_reg_loss

    return total_loss

def denormalize(image_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean = mean.to(device)
        std = std.to(device)
        image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]
        image_tensor = torch.clip(image_tensor, 0, 1)
        return image_tensor


def apply_nms(orig_boxes, orig_scores, iou_thresh=0.5):
    # Convert to torch tensors
    boxes = torch.as_tensor(orig_boxes, dtype=torch.float32)
    scores = torch.as_tensor(orig_scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = nms(boxes, scores, iou_thresh)
    
    # Convert back to numpy
    final_boxes = boxes[keep_indices].numpy()
    final_scores = scores[keep_indices].numpy()
    
    return final_boxes, final_scores

def visualize_batch_predictions(images, class_logits, box_coordinates, threshold=0.5):
    batch_size = images.size(0)
    for i in range(batch_size):
        image_tensor = images[i]  # Get the image tensor
        logits = class_logits[i]  # Get the logits for the current image
        boxes = box_coordinates[i].view(-1, 4)  # Get the bboxes for the current image
        
        scores = torch.sigmoid(logits).detach().cpu()  # Convert logits to probabilities and move to cpu
        scores, indices = torch.max(scores, dim=1)  # Get the max score for each box
        keep_boxes = scores > threshold  # Filter boxes by score threshold
        
        if keep_boxes.sum() > 0:  
            valid_boxes = boxes[keep_boxes].detach().cpu().numpy()
            valid_scores = scores[keep_boxes].detach().cpu().numpy()
            image_size = image_tensor.shape[1:]  # HxW

            # Apply NMS
            final_boxes, final_scores = apply_nms(valid_boxes, valid_scores)

            # Visualize the final predictions after NMS
            denormalized_image = denormalize(image_tensor).cpu()
            visualize_predictions(denormalized_image, final_boxes, final_scores, image_size)

def process_model_output(class_logits, bbox_coords, threshold=0.5):
    num_classes = 81
    num_boxes = 100  # Number of boxes per image
    batch_size = class_logits.shape[0]

    # Reshape logits to [batch_size, num_boxes, num_classes]
    reshaped_logits = class_logits.view(batch_size, num_boxes, num_classes)
    #print("reshaped logits", reshaped_logits)

    # Convert logits to probabilities using sigmoid
    probabilities = torch.sigmoid(reshaped_logits)
    #print("probs", probabilities)

    # Choose the label with the highest probability for each box
    max_probs, labels = torch.max(probabilities, dim=2)
    #print("max_probs", max_probs)

    # Initialize lists to store the results for each image
    all_selected_labels = []
    all_selected_boxes = []
    all_selected_scores = []

    # Reshape bbox_coords to match the logits shape for easy indexing
    reshaped_bbox_coords = bbox_coords.view(batch_size, num_boxes, 4)

    for i in range(batch_size):
        # Select boxes, labels, and scores for this image that exceed the threshold
        img_selected_indices = max_probs[i] >= threshold
        img_selected_labels = labels[i][img_selected_indices]
        img_selected_boxes = reshaped_bbox_coords[i][img_selected_indices]
        img_selected_scores = max_probs[i][img_selected_indices]

        # Add to the lists
        all_selected_labels.append(img_selected_labels)
        all_selected_boxes.append(img_selected_boxes)
        all_selected_scores.append(img_selected_scores)

    return all_selected_labels, all_selected_boxes, all_selected_scores

def main():
    
    batch_size = 64
    random_seed = 42
    num_classes = 81  # 1 COCO classes 
    num_boxes = 100  
    num_epochs = 12
    learning_rate = 0.005
    weight_decay = 0.0001
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    model = ObjectDetectionModel(num_classes=81, num_boxes=100)
    #checkpoint = torch.load("/content/object_detection_model_epoch_0.pth")
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(torch.load("/home/ps332/myViT/object_detection_model_final.pth"))

    eval_dataset = COCODataset(annotation_file="/home/ps332/myViT/coco_ann2017/annotations/instances_val2017.json",
                                image_dir= "/home/ps332/myViT/coco_val2017/val2017",
                                transform = transform)

    eval_loader = DataLoader(eval_dataset, batch_size = 1, shuffle = False, collate_fn = collate_fn )
    # To load the model, you will need to create an instance of the model class and then load the state dictionary into this instance:
    # model = ObjectDetectionModel(num_classes=num_classes, num_boxes=num_boxes)
    # model.load_state_dict(torch.load("/content/object_detection_model_final.pth"))

    model.to(device)
    model.eval()
    print("Running evaluation...")
    res = []
    id = 0

    with torch.no_grad():
        for imgs, targets in tqdm(eval_loader):
            images = imgs.to(device)
            batch_size = 1
            num_predicted_boxes = 100
            num_classes = 81
            threshold = 0.5

            class_logits, box_coordinates = model(images)
            selected_label, selected_boxes, selected_scores = process_model_output(class_logits, box_coordinates, threshold)
            # print("category id:", selected_label)
            # print("bbox:", selected_boxes)
            # print("score:", selected_scores)

            for i in range(batch_size):
                image_tensor = images[i]
                image_size = targets[i]["image_size"]  # wxh
                denormalized_image = denormalize(image_tensor).cpu()
                #visualize_predictions(denormalized_image, selected_boxes[i], selected_label[i], image_size, threshold)
                # print("image id:", targets[i]["image_id"])
                # print("category id:", selected_label[i])
                # print("bbox:", selected_boxes[i])
                # print("score:", selected_scores[i])

                for j in range(len(selected_label[i])):
                    if(selected_boxes[i][j][2]>selected_boxes[i][j][0] and selected_boxes[i][j][3]>selected_boxes[i][j][1]):
                        res.append({
                            "id": int(id),
                            "image_id": int(targets[i]["image_id"]),
                            "category_id": int(selected_label[i][j]),
                            "bbox": [
                                float(selected_boxes[i][j][0]*224),
                                float(selected_boxes[i][j][1]*224),
                                float(selected_boxes[i][j][2]*224-selected_boxes[i][j][0]*224),
                                float(selected_boxes[i][j][3]*224-selected_boxes[i][j][1]*224)
                            ],
                            "score": float(selected_scores[i][j])
                        })
                        id=id+1

    #print(res)
    output_file_path = "results.json"

    # Save the 'converted_data' to the JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(res, json_file,)

    print(f"Converted data has been saved to {output_file_path}")

    # cocoGt = COCO("/mnt/offload/mansi/VitObjDet/coco_val2017/instances_val2017.json")
    # cocoDt = cocoGt.loadRes(output_file_path)
    # cocoeval = COCOeval(cocoGt, cocoDt, "bbox")
    # cocoeval.evaluate()
    # cocoeval.accumulate()
    # cocoeval.summarize()

if __name__ == '__main__':
    main()            
