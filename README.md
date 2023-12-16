# ViTObjectDetection

This repository contains two models:
1. Our initial model with 'google/vit-base-patch16-224' as the backbone and our custom Object Detection Head.
2. Our secondary model (still work in progress) that incorporates DETR features for better object classification.

To run (1):
-Run coco.ipynb to gather datasets (both training and validation)
-ViTObjectDetectionTraining.py is the file where the model can be trained
-ObjectEval.py is the evaluation file that gathers results and turns them into results.json file
-Optional: You can run COCOEval.py but there are bugs that will lead to incorrect mAP metrics (which is why we show our losses)

To run (2):
-coco.ipynb is used again for datasets (both training and validation)
-DETRVersionofViTObjectDetection.py* is the file where the model can be trained
-ObjectEval.py is the evaluation file that gathers results and turns them into results.json file
-Optional: You can run COCOEval.py but there are bugs that will lead to incorrect mAP metrics (which is why we show our losses)

Note*: This is still a work in progress
