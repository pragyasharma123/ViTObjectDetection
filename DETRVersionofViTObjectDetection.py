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
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image
import warnings
warnings.filterwarnings('ignore')

model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class ObjectDetectionHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_queries, embedding_size):
        super(ObjectDetectionHead, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Single fully connected layer with BatchNorm
        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # Prediction heads for all queries
        self.classifier = nn.Linear(hidden_dim, num_classes * num_queries)
        self.box_regressor = nn.Linear(hidden_dim, 4 * num_queries)  # For bounding box coordinates

        # Activation function for bounding box coordinates
        self.box_regressor_activation = nn.Sigmoid()

    def forward(self, features):
        features = self.fc_layer(features)

        # Predict class labels and bounding boxes for each query
        class_logits = self.classifier(features).view(-1, self.num_queries, self.num_classes)
        box_coordinates = self.box_regressor_activation(self.box_regressor(features)).view(-1, self.num_queries, 4)

        return class_logits, box_coordinates



class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim=768):
        super(ObjectDetectionModel, self).__init__()

        self.hidden_dim = hidden_dim  # Define hidden_dim as an instance variable

        # Load ViT as the backbone
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        
        # Detection head
        self.head = ObjectDetectionHead(hidden_dim, num_classes, num_queries, hidden_dim)

    def forward(self, pixel_values):
        # Forward pass through the backbone
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state

        # Prepare object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, features.size(0), 1)

        # Transformer decoder to integrate features and queries
        decoded_features = self.transformer_decoder(query_embed, features.permute(1, 0, 2))

        # Prediction heads
        class_logits, box_coordinates = self.head(decoded_features.view(-1, self.hidden_dim))

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
        #print("target bboxes after conversion", bboxes)

        # Convert category IDs to names
        label_names = [self.coco_parser.id_to_name[label] for label in labels]
        
        # Map COCO category IDs to sequential indices
        labels = [self.coco_parser.cat_id_to_index[label] for label in labels]
        
        # Convert bboxes and labels to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": bboxes, "labels": label_names, "image_size": original_size}
        
        
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
    plt.savefig("/home/ps332/myViT/visualize_ground_truth.png")


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
        # Ensure that 'i' is within the valid range
        if i >= len(targets):
            continue  # Skip this iteration if 'i' is out of range

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

def main():
    
    batch_size = 64
    random_seed = 42
    num_classes = 81   
    num_boxes = 100  
    num_queries=100
    num_epochs = 5
    learning_rate = 0.005
    weight_decay = 0.0001
    torch.manual_seed(random_seed)
    random.seed(random_seed)
   
   # Initialize COCOParser
    coco_parser = COCOParser(anns_file="/home/ps332/myViT/coco_ann2017/annotations/instances_train2017.json",
                            imgs_dir="/home/ps332/myViT/coco_train2017/train2017")
    
    # Initialize COCODataset with the correct paths and transformation
    train_dataset = COCODataset(
        annotation_file="/home/ps332/myViT/coco_ann2017/annotations/instances_train2017.json",
        image_dir="/home/ps332/myViT/coco_train2017/train2017",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = ObjectDetectionModel(num_classes=81, num_queries=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # Device selection (CUDA GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    print("cache cleared")
    
    # move it to the device (e.g., GPU)
    model.to(device)
    print(model)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print("Epoch {} - Training starts".format(epoch))


        for batch_idx, (images, targets )in enumerate(tqdm(train_loader)):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            #visualize_bboxes(images[1], targets[1]) # debugging to make sure ground truth bboxes are correct
            
            optimizer.zero_grad()

            # Forward pass
            class_logits, box_coordinates = model(images)
            print("Predicted bbox format:", box_coordinates)

            # Compute loss using the simplified function
            loss = compute_loss(class_logits, box_coordinates, targets, coco_parser)

            print(f"loss:", loss)
            print(f"Average loss/length(train loader):", loss/len(train_loader))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # At the end of each epoch, save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f"/home/ps332/myViT/new_object_detection_model_epoch_{epoch}.pth")
        
        scheduler.step()

        print(f"Epoch {epoch}: Training Loss: {train_loss/len(train_loader)}")

        torch.save(model.state_dict(), "/home/ps332/myViT/new_object_detection_model_final.pth")



if __name__ == '__main__':
    main()            

############################################################################################################################################################
# To load the model, you will need to create an instance of the model class and then load the state dictionary into this instance:
#model = ObjectDetectionModel(num_classes=num_classes, num_boxes=num_boxes)
#model.load_state_dict(torch.load("/home/ps332/myViT/object_detection_model_final.pth"))
#model.to(device)  # Make sure to also call .to(device) if you're using a GPU

# If you also need to resume training from a checkpoint, you can load the optimizer state as well:
#checkpoint = torch.load("/home/ps332/myViT/object_detection_model_epoch_{epoch}.pth")
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']
