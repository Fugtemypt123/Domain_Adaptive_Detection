# Domain Adaptive Object Detection

In this project, we aim to solve the Domain Adaptive Object Detection (DAOD) task.

We use YOLO or Deformable DETR as the base detector. This framework is built upon the Deformable repository: https://github.com/fundamentalvision/Deformable-DETR. If you have limited GPU resources, YOLO detector(https://github.com/ultralytics/yolov5) may be a better choice, and you need to modify the framework.

## 1. Installation

### 1.1 Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

- Other requirements

  ```bash
  pip install -r requirements.txt
  ```

### 1.2 Compiling Deformable DETR CUDA operators

only for Deformable DETR

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## 2. Usage

### 2.1 Data preparation

We provide 3 benchmarks: 

- city2foggy: cityscapes dataset is used as source domain, and foggy_cityscapes(0.02) is used as target domain.
- sim2city: sim10k dataset is used as source domain, and cityscapes which only record AP of cars is used as target domain.
- city2bdd: cityscapes dataset is used as source domain, and bdd100k-daytime is used as target domain.

You can download the raw data from the official websites: [cityscapes](https://www.cityscapes-dataset.com/downloads/),  [foggy_cityscapes](https://www.cityscapes-dataset.com/downloads/),  [sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix), [bdd100k](https://bdd-data.berkeley.edu/). We provide the annotations that are converted into coco style, download from [here](https://drive.google.com/file/d/1LB0wK9kO3eW8jpR2ZtponmYWe9x2KSiU/view?usp=sharing) and organize the datasets and annotations as following:

```bash
[data_root]
└─ cityscapes
	└─ annotations
		└─ cityscapes_train_cocostyle.json
		└─ cityscapes_train_caronly_cocostyle.json
		└─ cityscapes_val_cocostyle.json
		└─ cityscapes_val_caronly_cocostyle.json
	└─ leftImg8bit
		└─ train
		└─ val
└─ foggy_cityscapes
	└─ annotations
		└─ foggy_cityscapes_train_cocostyle.json
		└─ foggy_cityscapes_val_cocostyle.json
	└─ leftImg8bit_foggy
		└─ train
		└─ val
└─ sim10k
	└─ annotations
		└─ sim10k_train_cocostyle.json
		└─ sim10k_val_cocostyle.json
	└─ JPEGImages
└─ bdd10k
	└─ annotations
		└─ bdd100k_daytime_train_cocostyle.json
		└─ bdd100k_daytime_val_cocostyle.json
	└─ JPEGImages
```

To use additional datasets, you can edit [datasets/coco_style_dataset.py] and add key-value pairs to `CocoStyleDataset.img_dirs` and `CocoStyleDataset.anno_files` .

### 2.2 Training and evaluation

As has been discussed in implementation details, we first perform `source_only` training which is trained standardly by labeled source domain. Then, we perform `teaching` which utilize a teacher-student framework.

For example, for `city2foggy` benchmark, first edit the files in `configs/def-detr-base/city2foggy/` to specify your own `DATA_ROOT` and `OUTPUT_DIR`, then run:

```bash
sh configs/def-detr-base/city2foggy/source_only.sh
sh configs/def-detr-base/city2foggy/teaching.sh
```

We use `tensorboard` to record the loss and results. Run the following command to see the curves during training: 

```bash
tensorboard --logdir=<YOUR/LOG/DIR>
```

To evaluate the trained model and get the predicted results, run:

```bash
sh configs/def-detr-base/city2foggy/evaluation.sh
```


## 3. Results and Report

You should conduct necessary experiments and report the results in a table. Here are examples:

**city2foggy**: cityscapes → foggy cityscapes(0.02)

| backbone | encoder layers | decoder layers | training stage   | AP@50 |
| -------- | -------------- | -------------- | ---------------- | ----- |
| resnet50 | 6              | 6              | source_only      | 29.5  |
| resnet50 | 6              | 6              | cross_domain_mae | 35.8  |
| resnet50 | 6              | 6              | MRT teaching     | 51.2  |

**sim2city**: sim10k → cityscapes(car only)

| backbone | encoder layers | decoder layers | training stage   | AP@50 |
| -------- | -------------- | -------------- | ---------------- | ----- |
| resnet50 | 6              | 6              | source_only      | 53.2  | 
| resnet50 | 6              | 6              | cross_domain_mae | 57.1  | 
| resnet50 | 6              | 6              | MRT teaching     | 62.0  | 

**city2bdd**: cityscapes → bdd100k(daytime)

| backbone | encoder layers | decoder layers | training stage   | AP@50 | 
| -------- | -------------- | -------------- | ---------------- | ----- | 
| resnet50 | 6              | 6              | source_only      | 29.6  | 
| resnet50 | 6              | 6              | cross_domain_mae | 31.1  | 
| resnet50 | 6              | 6              | MRT teaching     | 33.7  | 

