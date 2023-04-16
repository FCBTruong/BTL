# BTL

## Install yolo v5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

## Train model
python train.py --img 540 --batch 16 --epochs 10

## Detect 
python detect.py --weights runs/train/exp/weights/best.pt  --source ../datasets/coco128/images/test/img_test_01.jpg
