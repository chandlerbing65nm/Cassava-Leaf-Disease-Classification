![banner](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/Cassava_Leaf_Disease__nIdentification.png?raw=true)

![last commit](https://img.shields.io/github/last-commit/chandlerbing65nm/Cassava-Leaf-Disease-Classification) ![repo size](https://img.shields.io/github/repo-size/chandlerbing65nm/Cassava-Leaf-Disease-Classification) ![watchers](https://img.shields.io/github/watchers/chandlerbing65nm/Cassava-Leaf-Disease-Classification?style=social)

Image classification of cassava leaf diseases. This is based on the [Kaggle competition](https://www.kaggle.com/c/cassava-leaf-disease-classification) of the same title.

# Overview
![alt text](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/competition.png?raw=true)

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

# Requirements
- Linux/Windows/Mac
- Python 3
- Google Colab [GPU]

# Installation
You do not need to separately install any packages. The notebooks I attached are equipped with codes that will automatically download and install all necessary packages.

# Dataset
The data used can be downloaded [here](https://www.kaggle.com/c/cassava-leaf-disease-classification/data). The total size is 6.19Gb.

After downloading the data from kaggle, upload it into your google drive. 

There are Four classes of leaf diseases and One class of a healthy one. Here the names of the diseases and their classes.

1. [Cassava Bacterial Blight (CBB)](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/CBB.png)
2. [Cassava Brown Streak Disease (CBSD)](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/CBSD.png)
3. [Cassava Green Mottle (CGM)](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/CGM.png)
4. [Cassava Mosaic Disease (CMD)](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/CMD.png)
5. [Healthy](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/Healthy.png)

# Getting Started
Clone this repository and upload into your google drive.
Put the [Fmix](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/tree/main/image-fmix/FMix-master), [timm](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/tree/main/pytorch-image-models/pytorch-image-models-master) and the [dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification/data) into your Gdrive directory:
> Kaggle-Projects/Cassava-Leaf-Disease/input

Make sure you copy your project folder to colab directory so you don't casually edit your drive directory. This is already in the notebook, don't worry.
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

!cp -r /content/drive/MyDrive/Kaggle-Projects/Cassava-Leaf-Disease /content
%cd Cassava-Leaf-Disease
```

## Training
Open [ViT-Training.ipynb](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/ViT_Training.ipynb) and click open in colab.

Manually run each cell until you reach the cell \***Helper Functions (perform this twice)**\*. You should perform this twice since it will be an error in the first run. Google Colab had a problem with the \***cv.imread()**\* and \***plt.imread()**\* functions.

This is the code in the cell.
```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(pathw):
    im_bgr = plt.imread(pathw)
    im_rgb = im_bgr
    return im_rgb

img = get_img('input/cassava-leaf-disease-classification/train_images/1000015157.jpg')
plt.imshow(img)
plt.show()
```

The training loop at \***Main Training Loop**\* have 5 folds with 9 epochs each fold. Using NVIDIA P100 GPU, the approximate training time for the 5 folds is +/-1Hour.

After the training, inspect the \***validation multi-class accuracy**\* of each fold. Then, save the epoch 6-9 of the highest accuracy fold into your GDrive.

Save in your Gdrive directory:
> Kaggle-Projects/Cassava-Leaf-Disease/output


In my run, the highest accuracy that I got is from fold 2. Here is my training accuracy at fold 2:
- [x] validation multi-class accuracy: 0.8605


## Inference
Open [ViT-Inference.ipynb](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/ViT_Inference.ipynb) and click open in colab.

Again, manually run each cell until you reach the cell \***Helper Functions (perform this twice)**\*. You should perform this twice since it will be an error in the first run. Google Colab had a problem with the \***cv.imread()**\* and \***plt.imread()**\* functions.

In the \***Main Inference Loop**\* cell, be aware that you should replace this line:
```python
model.load_state_dict(torch.load('output/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
```

with the directory of which you saved your model checkpoints from training. But if you already placed the model checkpoints in the output folder, you're good to go.

Alternatively, you can run my [notebook in kaggle](https://www.kaggle.com/chandlertimm/vit-inference). Click edit and select GPU in the resources. You can run-all automatically, no errors will be encountered unlike in colab.

My inference loss and accuracy are:
- [x] validation loss: 0.21047
- [x] validation accuracy: 0.94017

Take note that the test set of this competition is hidden, so you need to submit your results after the inference. In Kaggle, it will take 1-2 hours.

# Results
In my submission, I got a test set accuracy of 0.8838 or 88.38%.

![alt text](https://github.com/chandlerbing65nm/Cassava-Leaf-Disease-Classification/blob/main/images/results.jpg?raw=true)
