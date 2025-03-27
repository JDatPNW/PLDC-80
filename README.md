# PLDC-80: A Combined Augmented Hybrid Plant Leaf Disease Classification Benchmarking Dataset

![Banner.](/banner.png)



> PLDC-80 Dataset methodology repository. This repository contains all the materials required to download and build the PLDC-80 dataset for plant leaf disease classification benchmarking. In this Readme file you can find a detailed step by step guide on how to create said PLDC-80 dataset with files that automate most of the process, with only few steps such as downloading, copying the right files into the right location and naming the folders accordingly to allows the code to function. To allow the code provided to work the steps need to be followec closely, since the code requires the specific names and directories to be used. If any of the steps fail or if you prefer to create the dataset, you can also do each step manually, by following the steps provided in the paper, the guide below, and by looking at the provided code.

## 1. Dataset list with Download links

| Dataset      | Link |
|--------------|------|
| cassava      |[cassava](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)|
| cds          |[cds](https://osf.io/s6ru5/)|
| diaMOS       |[diaMOS](https://zenodo.org/records/5557313)|
| fgvc8        |[fgvc8](https://www.kaggle.com/c/plant-pathology-2021-fgvc8)|
| paddy        |[paddy](https://www.kaggle.com/competitions/paddy-disease-classification)|
| pdd271       |[pdd271](https://github.com/liuxindazz/PDD271)|
| plantVillage |[plantVillage](https://github.com/spMohanty/PlantVillage-Dataset)|
| sms          |[sms](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)|
| sugar        |[sugar](https://data.mendeley.com/datasets/9twjtv92vk/1)|


## 2. Class by class preperation

### sms (Strawberry)
1. [Go to Dataset webpage](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
2. Extract zip
3. rename extracted folder to `sms`
4. move to `./pre_clean_data/sms`
5. run `sms_prep.py`
6. processed data is now available in `./data/sms`
7. remove the classes called `anthracnose_fruit_rot`, `blossom_blight`, `gray_mold`, and `powdery_mildew_fruit`
8. can delete sms zip file and `sms` folder in `./pre_clean_data/` now


<details>
<summary>Expected results</summary>

- 3 classes
- 1,583 images in total
- Structure:

```properties
data
└──────sms
        │
        └─── angular_leafspot
        │
        └─── leaf_spot
        │
        └─── powdery_mildew_leaf
```

</details>

### cassava
1. [Go to Dataset webpage](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
2. Login to Kaggle and join competition
3. Download dataset
4. extract zip
5. rename extracted folder to `cassava`
6. move to `./pre_clean_data/sms`
7. run `cassava_prep.py`
8. processed data is now available in `./data/cassava`
9. can delete cassava zip file and `cassava` folder in `./pre_clean_data/` now


<details>
<summary>Expected results</summary>

- 5 classes
- 21,397 images in total
- Structure:

```properties
data
└────cassava
        │
        └─── Cassava Bacterial Blight (CBB)
        │
        └─── Cassava Brown Streak Disease (CBSD)
        │
        └─── Cassava Green Mottle (CGM)
        │
        └─── Cassava Mosaic Disease (CMD)
        │
        └─── Healthy
```

</details>

### fgvc8
1. [Go to Dataset webpage](https://www.kaggle.com/c/plant-pathology-2021-fgvc8)
2. Login to Kaggle and join competition
3. Download dataset
4. extract zip
5. rename extracted folder to `fgvc8`
6. move to `./pre_clean_data/fgvc8`
7. run `fgvc8_prep.py`
8. processed data is now available in `./data/fgvc8`
9. can delete fgvc8 zip file and `fgvc8` folder in `./pre_clean_data/` now

 
<details>
<summary>Expected results</summary>

- 5 classes
- 15,675 images in total
- Structure:

```properties
data
└─────fgvc8
        │
        └─── frog_eye_leaf_spot
        │
        └─── healthy
        │
        └─── powdery_mildew
        │
        └─── rust
        │
        └─── scab
```

</details>


### cds 
1. [Go to Dataset webpage](https://osf.io/s6ru5/)
2. Navigate to `files` &rarr; `CD&S` &rarr; `Dataset_Original`  
3. Download dataset (`Download this folder`)
4. extract zip
5. rename extracted folder to `cds`
6. move to `./pre_clean_data/cds`
7. run `cds_prep.py`
8. processed data is now available in `./data/cds`
9. can delete cds zip file and `cds` folder in `./pre_clean_data/` now

<details>
<summary>Expected results</summary>

- 5 classes
- 1,571 images in total
- Structure:

```properties
data
└─────cds
        │
        └─── gls
        │
        └─── nlb
        │
        └─── nls
```

</details>


### plantVillage 
1. [Go to Dataset webpage](https://github.com/spMohanty/PlantVillage-Dataset)
2. Download Dataset
3. extract zip
4. Navigate to `raw` &rarr; `color` 
5. rename the color folder to  `plantVillage`
6. move to that foplder to `./pre_clean_data/plantVillage`
7. run `plantVillage_prep.py`
8. processed data is now available in `./data/plantVillage`
9.  can delete plantVillage zip file and `plantVillage` folder in `./pre_clean_data/` now

<details>
<summary>Expected results</summary>

- 37 classes
- 54,153 images in total
- Structure:

```properties
data
└────plantVillage
        │
        └─── Apple___Apple_scab
        │
        └─── Apple___Black_rot
        │
        └─── Apple___Cedar_apple_rust
        │
        └─── Apple___healthy
        │
        └─── Blueberry___healthy
        │
        └─── Cherry_(including_sour)___healthy
        │
        └─── Cherry_(including_sour)___Powdery_mildew
        │
        └─── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
        │
        └─── Corn_(maize)___Common_rust_
        │
        └─── Corn_(maize)___healthy
        │
        └─── Corn_(maize)___Northern_Leaf_Blight
        │
        └─── Grape___Black_rot
        │
        └─── Grape___Esca_(Black_Measles)
        │
        └─── Grape___healthy
        │
        └─── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
        │
        └─── Orange___Haunglongbing_(Citrus_greening)
        │
        └─── Peach___Bacterial_spot
        │
        └─── Peach___healthy
        │
        └─── Pepper,_bell___Bacterial_spot
        │
        └─── Pepper,_bell___healthy
        │
        └─── Potato___Early_blight
        │
        └─── Potato___Late_blight
        │
        └─── Raspberry___healthy
        │
        └─── Soybean___healthy
        │
        └─── Squash___Powdery_mildew
        │
        └─── Strawberry___healthy
        │
        └─── Strawberry___Leaf_scorch
        │
        └─── Tomato___Bacterial_spot
        │
        └─── Tomato___Early_blight
        │
        └─── Tomato___healthy
        │
        └─── Tomato___Late_blight
        │
        └─── Tomato___Leaf_Mold
        │
        └─── Tomato___Septoria_leaf_spot
        │
        └─── Tomato___Spider_mites Two-spotted_spider_mite
        │
        └─── Tomato___Target_Spot
        │
        └─── Tomato___Tomato_mosaic_virus
        │
        └─── Tomato___Tomato_Yellow_Leaf_Curl_Virus
```

</details>


### diaMOS
1. [Go to Dataset webpage](https://zenodo.org/records/5557313)
2. Download dataset
3. extract zip
4. rename extracted folder to `diaMOS`
5. navigate to `disMOS/Pear/leaves`
6. move that `leaves` folder to `./pre_clean_data` and name it `diaMOS`
7. run `diaMOS_prep.py`
8. processed data is now available in `./data/diaMOS`
9. can delete diaMOS zip file and `diaMOS` folder in `./pre_clean_data/` now

 
<details>
<summary>Expected results</summary>

- 2 classes
- 5,818 images in total
- Structure:

```properties
data
└────diaMOS
        │
        └─── slug
        │
        └─── spot
```

</details>


### paddy
1. [Go to Dataset webpage](https://www.kaggle.com/competitions/paddy-disease-classification)
2. join competition
3. download dataset
4. extract zip
5. rename extracted folder to `paddy`
6. move that `paddy` folder to `./pre_clean_data`
7. run `paddy_prep.py`
8. processed data is now available in `./data/paddy`
9.  can delete paddy zip file and `paddy` folder in `./pre_clean_data/` now

 
<details>
<summary>Expected results</summary>

- 10 classes
- 10,407 images in total
- Structure:

```properties
data
└────paddy
        │
        └─── bacterial_leaf_blight
        │
        └─── bacterial_leaf_streak
        │
        └─── bacterial_panicle_blight
        │
        └─── blast
        │
        └─── brown_spot
        │
        └─── dead_heart
        │
        └─── downy_mildew
        │
        └─── hispa
        │
        └─── normal
        │
        └─── tungro
```

</details>


### sugar
1. [Go to Dataset webpage](https://data.mendeley.com/datasets/9twjtv92vk/1)
2. download the data `Download All 745MB`
3. extract zip
4. navigate into the extracted folder and then `./Sugarcane Leaf Image Dataset/`
5. inside that folder unzip the `Healthy Leaves.zip` and the `Diseases.zip` (not the `Dried Leaves.zip`)
6. create a folder called `sugar` in `./pre_clean_data` and move both unzipped folders into the `sugar` folder
7. run `sugar_prep.py`
8. processed data is now available in `./data/sugar`
9.  can delete sugar zip file and `sugar` folder in `./pre_clean_data/` now

 
<details>
<summary>Expected results</summary>

- 10 classes
- 6,405 images in total
- Structure:

```properties
data
└────sugar
        │
        └─── Banded Chlorosis
        │
        └─── Brown Spot
        │
        └─── BrownRust
        │
        └─── Grassy shoot
        │
        └─── Healthy Leaves
        │
        └─── Pokkah Boeng
        │
        └─── Sett Rot
        │
        └─── smut
        │
        └─── Viral Disease
        │
        └─── Yellow Leaf
```

</details>

### pdd271
1. [Go to Dataset webpage](https://github.com/liuxindazz/PDD271)
2. at the bottom of the ReadMe is the link to the GoogleDrive with the image data
3. open the link and download the data
4. extract zip
5. navigate into the extracted folder and rename the `Sample` folder to `pdd271`
6. move that folder to `./pre_clean_data`
7. run `pdd271_prep.py`
8.  processed data is now available in `./data/pdd271`
9.  can delete pdd271 zip file and `pdd271` folder in `./pre_clean_data/` now

 
<details>
<summary>Expected results</summary>

- 10 classes
- 7,555 images in total
- Structure:

```properties
data
└────paddy
        │
        └─── leek_gray_mold_disease_339
        │
        └─── leek_hail_damage_338
        │
        └─── Mung_bean_brown_spot_246
        │
        └─── radish_black_spot_disease_297
        │
        └─── radish_mosaic_virus_disease_295
        │
        └─── radish_wrinkle_virus_disease_293
        │
        └─── Soybean_downy_mildew_135
        │
        └─── Sweet_potato_healthy_leaf_220
        │
        └─── Sweet_potato_magnesium_deficiency_227
        │
        └─── Sweet_potato_sooty_mold_224
```

</details>


### After downloading and preparing datasets
Delete the `pre_clean_data` with all its contents.

The classes deleted during this process are:
| source dataset| class name                            |
|---------------|---------------------------------------|
| diaMOS        | Pear Healthy                          |
| diaMOS        | Pear Curl                             |
| fgvc8         | Apple Powdery Mildew Complex          |
| fgvc8         | Apple Rust Complex                    |
| fgvc8         | Apple Rust Frog Eye Leaf Spot         |
| fgvc8         | Apple Frog Eye Leaf Spot Complex      |
| fgvc8         | Apple Scab Frog Eye Leaf Spot Complex |
| fgvc8         | Complex                               |
| fgvc8         | Scab Frog Eye Leaf Spot               |
| PlantVillage  | Potato Healthy                        |


<details>
<summary>Structure</summary>

- 85 classes
- 124,564  images in total
 
</details>




## 3. Rename, merge and combine

If the folder strucutre as above is followed, you can simply run the `pre_process_all.py` which will merge all folders in `./data` into `./data_merged` and rename the folders acroding to the original dataset as well as merge the 5 necessary class pairs.

Merged Classes:

| Dataset 1 | Class 1              | Dataset 2    | Class 2                             |
|-----------|----------------------|--------------|-------------------------------------|
| fgvc8     | Healthy              | PlantVillage | Apple Healthy                       |
| fgvc8     | Rust                 | PlantVillage | Cedar Apple Rust                    |
| fgvc8     | Scab                 | PlantVillage | Apple Apple Scab                    |
| cds       | Gray Leaf Spot       | PlantVillage | Cercospora Leaf Spot Gray Leaf Spot |
| cds       | Northern Leaf Blight | PlantVillage | Northern Leaf Blight                |


<details>
<summary>Structure</summary>

- 80 classes
- 124,564  images in total
 
</details>

The `./data` folder should now contain no more images, and can be deleted

## 4. Train-Test split

- run the `train_test_split.py` which will split images 80-20 into the data_merged_split folder. 
- Make sure to do this before augmentation 

## 5. Augmentation

- run the `augment_imges.py` file which will augment all the `train` data and move the `test` data without modifying it.

## 6. Cut-off

- finally, the last step is to cutoff the dataset at 3,500 image per class for perfect balancing. To to this simply run the `cutoff.py` file.
- Now the final result should look like:

---

> <ins>**This has now created the final dataset file in the `PLDC80` folder.** </ins>
> 
> <ins>**All other folders are not part of the dataset and can be disregarded/deleted.** </ins>
> 
> <ins>**Use the data contained in `PLDC80` for training/benchmarking.**</ins>

---


<details>
<summary>Final Structure</summary>

- 80 classes
- 280,000  images in total
- 24,507 test images

```properties
data
└───train/test
        │
        └─── cassava_cassava_Cassava Bacterial Blight (CBB)
        │
        └─── cassava_cassava_Cassava Brown Streak Disease (CBSD)
        │
        └─── cassava_cassava_Cassava Green Mottle (CGM)
        │
        └─── cassava_cassava_Cassava Mosaic Disease (CMD)
        │
        └─── cassava_cassava_Healthy
        │
        └─── cds_corn_Gray Leaf Spot_AND_plantvillage_Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
        │
        └─── cds_corn_Northern Leaf Blight_AND_plantvillage_Corn_(maize)___Northern_Leaf_Blight
        │
        └─── cds_corn_Northern Leaf Spot
        │
        └─── diamos_pear_slug
        │
        └─── diamos_pear_spot
        │
        └─── fgvc8_apple_frog_eye_leaf_spot
        │
        └─── fgvc8_apple_healthy_AND_plantvillage_Apple___healthy
        │
        └─── fgvc8_apple_powdery_mildew
        │
        └─── fgvc8_apple_rust_AND_plantvillage_Apple___Cedar_apple_rust
        │
        └─── fgvc8_apple_scab_AND_plantvillage_Apple___Apple_scab
        │
        └─── paddy_rice_bacterial_leaf_blight
        │
        └─── paddy_rice_bacterial_leaf_streak
        │
        └─── paddy_rice_bacterial_panicle_blight
        │
        └─── paddy_rice_blast
        │
        └─── paddy_rice_brown_spot
        │
        └─── paddy_rice_dead_heart
        │
        └─── paddy_rice_downy_mildew
        │
        └─── paddy_rice_hispa
        │
        └─── paddy_rice_normal
        │
        └─── paddy_rice_tungro
        │
        └─── Soybean_downy_mildew_135
        │
        └─── pdd271_Mung_bean_brown_spot_246
        │
        └─── pdd271_Sweet_potato_healthy_leaf_220
        │
        └─── pdd271_Sweet_potato_magnesium_deficiency_227
        │
        └─── pdd271_Sweet_potato_sooty_mold_224
        │
        └─── pdd271_leek_gray_mold_disease_339
        │
        └─── pdd271_leek_hail_damage_338
        │
        └─── pdd271_radish_black_spot_disease_297
        │
        └─── pdd271_radish_mosaic_virus_disease_295
        │
        └─── pdd271_radish_wrinkle_virus_disease_293
        │
        └─── plantvillage_Apple___Black_rot
        │
        └─── plantvillage_Blueberry___healthy
        │
        └─── plantvillage_Cherry_(including_sour)___Powdery_mildew
        │
        └─── plantvillage_Cherry_(including_sour)___healthy
        │
        └─── plantvillage_Corn_(maize)___Common_rust_
        │
        └─── plantvillage_Corn_(maize)___healthy
        │
        └─── plantvillage_Grape___Black_rot
        │
        └─── plantvillage_Grape___Esca_(Black_Measles)
        │
        └─── plantvillage_Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
        │
        └─── plantvillage_Grape___healthy
        │
        └─── plantvillage_Orange___Haunglongbing_(Citrus_greening)
        │
        └─── plantvillage_Peach___Bacterial_spot
        │
        └─── plantvillage_Peach___healthy
        │
        └─── plantvillage_Pepper,_bell___Bacterial_spot
        │
        └─── plantvillage_Pepper,_bell___healthy
        │
        └─── plantvillage_Potato___Early_blight
        │
        └─── plantvillage_Potato___Late_blight
        │
        └─── plantvillage_Raspberry___healthy
        │
        └─── plantvillage_Soybean___healthy
        │
        └─── plantvillage_Squash___Powdery_mildew
        │
        └─── plantvillage_Strawberry___Leaf_scorch
        │
        └─── plantvillage_Strawberry___healthy
        │
        └─── plantvillage_Tomato___Bacterial_spot
        │
        └─── plantvillage_Tomato___Early_blight
        │
        └─── plantvillage_Tomato___Late_blight
        │
        └─── plantvillage_Tomato___Leaf_Mold
        │
        └─── plantvillage_Tomato___Septoria_leaf_spot
        │
        └─── plantvillage_Tomato___Spider_mites Two└───spotted_spider_mite
        │
        └─── plantvillage_Tomato___Target_Spot
        │
        └─── plantvillage_Tomato___Tomato_Yellow_Leaf_Curl_Virus
        │
        └─── plantvillage_Tomato___Tomato_mosaic_virus
        │
        └─── plantvillage_Tomato___healthy
        │
        └─── sms_strawberry_angular_leafspot
        │
        └─── sms_strawberry_leaf_spot
        │
        └─── sms_strawberry_powdery_mildew_leaf
        │
        └─── sugar_cane_Banded Chlorosis
        │
        └─── sugar_cane_Brown Spot
        │
        └─── sugar_cane_BrownRust
        │
        └─── sugar_cane_Grassy shoot
        │
        └─── sugar_cane_Healthy Leaves
        │
        └─── sugar_cane_Pokkah Boeng
        │
        └─── sugar_cane_Sett Rot
        │
        └─── sugar_cane_Viral Disease
        │
        └─── sugar_cane_Yellow Leaf
        │
        └─── sugar_cane_smut
```
</details>

## 7. Sample usage

This code shows a simple use case of how one could load the data in `tensorflow`, splitting the train data a further 80-20 into train and val while keeping the un-augmented test data intact.

```python
data_directory = "./PLDC80/"

# Loads training portion of dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory + "train",
    label_mode='categorical', 
    validation_split=0.2,
    subset="training",
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

# Loads validation portion of dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory + "train",
    validation_split=0.2,
    subset="validation",
    label_mode='categorical',
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

# Loads training portion of dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory + "test",
    label_mode='categorical', 
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)
```

after which the output should look like this:
```
Found 280000 files belonging to 80 classes.
Using 224000 files for training.
Found 280000 files belonging to 80 classes.
Using 56000 files for validation.
Found 24507 files belonging to 80 classes.
```

## 8. Summary/Recap/Quick Instruction

1. Download all the datasets and prepare them as instructed above
2. run all the `*_prep.py` files
3. run `pre_process_all.py`
4. run `train_test_split.py`
5. run `augment_images.py`
6. run `cutoff.py`

## Citation
If you use this dataset please cite:

   ```ini
   placeholder for bibtex    
   ```

### Python prerequisites
This code was tested with these version. Other versions might work too but were not tested.

| package       | version                               |
|---------------|---------------------------------------|
| python        | 3.11.5                                |
| opencv-python | 4.8.1.78                              |
| numpy         | 1.26.2                                |
| pandas        | 2.2.2                                 |

### Image rights
Not all datasets included in PLDC-80 permit redistribution. Therefore, we only share the methods for creating PLDC-80 from the sub-datasets without distributing any images. All image rights remain with their original owners.

### Citations for individual datasets used in PLDC-80:
We encourage the citation of all sub-datasets when using PLDC-80. All references are listed below. Reading each of these papers will also give much more detailed insight into each of the 9 datasets.

1. cassava
   ```ini
   @article{mwebaze2019icassava,
    title={iCassava 2019 fine-grained visual categorization challenge},
    author={Mwebaze, Ernest and Gebru, Timnit and Frome, Andrea and Nsumba, Solomon and Tusubira, Jeremy},
    journal={arXiv preprint arXiv:1908.02900},
    year={2019},
    doi={10.48550/arXiv.1908.02900}
    } 
   ```

2. cds
   ```ini
    @article{ahmad2021cd,
    title={CD\&S dataset: Handheld imagery dataset acquired under field conditions for corn disease identification and severity estimation},
    author={Ahmad, Aanis and Saraswat, Dharmendra and Gamal, Aly El and Johal, Gurmukh},
    journal={arXiv preprint arXiv:2110.12084},
    year={2021},
    doi={10.48550/arXiv.2110.12084}
    }   
    ```

3. diaMOS
   ```ini
    @article{fenu2021diamos,
    title={DiaMOS plant: A dataset for diagnosis and monitoring plant disease},
    author={Fenu, Gianni and Malloci, Francesca Maridina},
    journal={Agronomy},
    volume={11},
    number={11},
    pages={2107},
    year={2021},
    publisher={MDPI},
    doi={10.3390/agronomy11112107}
    }
    ```
    
4. fgvc8
   ```ini
    @article{thapa2020plant,
    title={The Plant Pathology Challenge 2020 data set to classify foliar disease of apples},
    author={Thapa, Ranjita and Zhang, Kai and Snavely, Noah and Belongie, Serge and Khan, Awais},
    journal={Applications in plant sciences},
    volume={8},
    number={9},
    pages={e11390},
    year={2020},
    publisher={Wiley Online Library},
    doi={10.1002/aps3.11390}
    }
    ```

5. paddy
   ```ini
    @inproceedings{petchiammal2023paddy,
    title={Paddy doctor: A visual image dataset for automated paddy disease classification and benchmarking},
    author={Petchiammal and Kiruba, Briskline and Murugan and Arjunan, Pandarasamy},
    booktitle={Proceedings of the 6th Joint International Conference on Data Science \& Management of Data (10th ACM IKDD CODS and 28th COMAD)},
    pages={203--207},
    year={2023},
    doi={10.1145/3570991.35709}
    }    
    ```

6. pdd271
   ```ini
    @article{liu2021plant,
    title={Plant disease recognition: A large-scale benchmark dataset and a visual region and loss reweighting approach},
    author={Liu, Xinda and Min, Weiqing and Mei, Shuhuan and Wang, Lili and Jiang, Shuqiang},
    journal={IEEE Transactions on Image Processing},
    volume={30},
    pages={2003--2015},
    year={2021},
    publisher={IEEE},
    doi={10.1109/TIP.2021.3049334}
    }
    ```

7. plantVillage
   ```ini
    @article{mohanty2016using,
    title={Using deep learning for image-based plant disease detection},
    author={Mohanty, Sharada P and Hughes, David P and Salathé, Marcel},
    journal={Frontiers in plant science},
    volume={7},
    pages={215232},
    year={2016},
    publisher={frontiers},
    doi={10.3389/fpls.2016.01419}
    }
   ```

8. sms
   ```ini
    @article{afzaal2021instance,
    title={An instance segmentation model for strawberry diseases based on mask R-CNN},
    author={Afzaal, Usman and Bhattarai, Bhuwan and Pandeya, Yagya Raj and Lee, Joonwhoan},
    journal={Sensors},
    volume={21},
    number={19},
    pages={6565},
    year={2021},
    publisher={MDPI},
    doi={10.3390/s21196565}
    }
   ```

9. sugar
    ```ini
    @article{thite2024sugarcane,
    title={Sugarcane leaf dataset: A dataset for disease detection and classification for machine learning applications},
    author={Thite, Sandip and Suryawanshi, Yogesh and Patil, Kailas and Chumchu, Prawit},
    journal={Data in Brief},
    volume={53},
    pages={110268},
    year={2024},
    publisher={Elsevier},
    doi={10.1016/j.dib.2024.110268}
    }
    ```