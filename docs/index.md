This [repository](https://github.com/cheeweeng/Training-and-Fine-tuning-an-image-recognition-and-classification-model) contains the 
python codes to develop an image classification model utilizing a pre-trained MobileNetV2 convolutional base and submitted in June 2026 for the Deep Learning for Image Recognition course assignment 
in Ngee Ann Polytechnic. 

# Overview  
The Deep Learning (DLIR) assignment for the Specialist Diploma in Applied Generative AI tasks students with a comprehensive computer vision project.  

### The Problem and Objective   
The core objective of this assignment is to build and evaluate deep learning image classification models capable of recognizing and categorizing ten different types of food. Accurately classifying images requires the application of neural networks to process raw image data and map it to specific food classes. The assignment specifies a structured, four-step approach to properly address this image classification problem:  
1.	Data Loading and Preprocessing:  
A dataset which is pre-divided into specific subsets: a training set containing 750 images per food type, a validation set with 200 images per food type, and a testing set with 50 images per food type is provided. The provided training, validation, and test image datasets were loaded into Jupyter Notebook.  
2.	Model Development:   
The core of the approach involves developing at least two distinct image classification models. The first model must be built entirely from scratch, utilizing standard 2D convolutional (conv2D) and dense layers. The second model must leverage the power of pre-trained networks. Both approaches are required to follow a universal machine learning workflow: establishing a baseline model, scaling it up until overfitting occurs, applying necessary regularization techniques, and rigorously tuning hyperparameters. Throughout this training phase, model performance curves must be carefully recorded for later analysis.  
3.	Model Evaluation:   
After the training phase is complete, both models must be evaluated against the unseen testing dataset. The performance of both models were compared to determine which approach yields superior results during testing and recommend the single best model based.
4.	Real-World Prediction:   
To test the practical robustness of the recommended best model, three novel food images were sourced directly from the internet and fed into the selected model to verify if it can accurately classify real-world data outside of the provided dataset.  

Ultimately, this structured approach forms the basis of the required final deliverables, which include a set of final presentation PowerPoint slides, a comprehensive individual report (Word document) detailing the methodology, and a Jupyter notebook (multiple notebooks acceptable) containing the functional code.  

This github page documents the full pipeline implemented in — data loading, preprocessing, model architecture, hyperparameter tuning across three iterations, evaluation results, and a final recommendation.  

---
# 1. Overview of Approach  

Three progressively refined variants of the same transfer-learning model were built:  

Model	Description  

Model 2a	Feature extraction — frozen MobileNetV2 base, features pre-computed once, no augmentation  

Model 2b	Feature extraction + data augmentation — frozen base trained end-to-end on augmented images  

Model 2c	Fine-tuning — top blocks of MobileNetV2 (blocks 13–16 + `Conv_1`) unfrozen and trained at a very low learning rate 

All three share the same classifier head and build on top of the same pre-trained convolutional base.  

---
# 2. Data Loading
Images were organised on disk in the standard Keras directory-per-class layout:
```
/food
├── train/        (10 class subfolders)
├── validation/    (10 class subfolders)
└── test/        (10 class subfolders)
```
Loaded using `keras.utils.image_dataset_from_directory':  

Split	Images found	Classes

Train	7,500	10  

Validation	2,000	10  

Test	500	10  

Loading configuration:  
Parameter	Value  

`IMG_SIZE`	(224, 224) — required input size for MobileNetV2  

`BATCH_SIZE`	20 

`NUM_CLASSES`	10  

A quick sanity check on a sample batch confirmed integer labels `0–9` (sparse / non-one-hot encoding), which dictated the choice of loss function later (`sparse_categorical_crossentropy`).  

---
# 3. Data Pre-processing
3.1 Pixel pre-processing

All images were passed through `mobilenet_v2.preprocess_input()`, which scales pixel values into the range MobileNetV2 expects (its ImageNet pre-training range), rather than the simple 0–1 rescaling used for CNNs trained from scratch.

3.2 Data Augmentation (Models 2b & 2c only)

To reduce overfitting once the model moved from pre-computed features to end-to-end training, the following on-the-fly augmentation layers were applied only to the training set:


`RandomFlip`	`"horizontal"`	Random left-right mirroring

`RandomRotation`	±10% of 360°	Random small rotations

`RandomZoom`	±20% (height & width)	Random zoom in/out

`RandomTranslation`	±20% (height & width)	Random shifting


3.3 Feature pre-computation (Model 2a only)

For the first iteration, instead of running every training image through MobileNetV2 on every epoch, features were pre-computed once by passing all images through the frozen convolutional base and caching the resulting `(7, 7, 1280)` feature maps:

Output	Shape	Meaning

`train_features`	(7500, 7, 7, 1280)	One 7×7×1280 feature map per training image

`train_labels`	(7500,)	One integer label (0–9) per image

This made training the lightweight classifier head dramatically faster, since the 2.2M frozen MobileNetV2 parameters never need to be recomputed across epochs. This shortcut is only valid as long as the base is frozen — it was abandoned once augmentation and fine-tuning were introduced, since augmented images differ every epoch and the base itself starts updating.

---
# 4. Model Architecture
All three models share the same backbone and classifier head structure:
<img width="1095" height="93" alt="finetune_layers" src="https://github.com/user-attachments/assets/42b2e714-37b5-4a29-9d73-9edf57de1ecd" />

Convolutional base:
```python
conv_base = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,     # discard ImageNet's 1000-class head
    weights='imagenet'     # pre-trained weights
)
conv_base.trainable = False  # frozen for Models 2a & 2b
```
Classifier head (added on top of the base):
```python
model = models.Sequential([
    conv_base,                                    # or pre-computed features as Input
    layers.GlobalAveragePooling2D(),              # (7,7,1280) -> (1280,)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
```
Design choices:  

<table>
  <tr>
    <td width="45%">
      <img src="https://github.com/user-attachments/assets/cbf19836-b4c1-47ad-a555-aeab2c42935e"
           alt="GlobalAveragePooling2D diagram"
           width="500">
    </td>
    <td width="55%" valign="top">

### GlobalAveragePooling2D over Flatten

`GlobalAveragePooling2D` collapses the `7×7×1280` feature map into a `1280`-length vector instead of producing a much larger flattened vector.

This approach:

- Keeps the number of parameters in the subsequent Dense layer low.
- Reduces the risk of overfitting.
- Improves computational efficiency and memory usage.
- Is the standard choice for transfer learning with modern CNN architectures 
    </td>
  </tr>
</table>  

Dropout (0.3) before the output layer acts as a regulariser against overfitting once training moved beyond the simple feature-extraction baseline.
Softmax output over 10 units, paired with `sparse_categorical_crossentropy`, matches the integer-encoded labels produced by `image_dataset_from_directory`.

---
# 5. Hyperparameters & Training Iterations

### Model 2a — Feature Extraction (frozen base, no augmentation)  

<table>
<tr>
<td width="60%" valign="top">
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/48693b42-f56d-451c-bc8d-492c22e1831d" />
</td>
<td width="40%" valign="top">

<table>
<tr><th>Hyperparameter</th><th>Value</th></tr>
<tr><td>Optimizer</td><td>Adam</td></tr>
<tr><td>Learning rate</td><td>2e-5</td></tr>
<tr><td>Loss</td><td><code>sparse_categorical_crossentropy</code></td></tr>
<tr><td>Metric</td><td>accuracy</td></tr>
<tr><td>Epochs</td><td>30</td></tr>
<tr><td>Batch size</td><td>20</td></tr>
<tr><td>Trainable base</td><td>No (fully frozen)</td></tr>
</table>

</td>
</tr>
</table>

Result:   
Training accuracy climbed to ~84%; validation accuracy plateaued around 80–81% by epoch 30. Validation loss flattened near 0.58 while training loss kept falling — a sign of mild overfitting, as the gap between training and validation accuracy widened over time.

### Model 2b — Feature Extraction + Data Augmentation (frozen base)

<table>
<tr>
<td width="60%" valign="top">
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/6a3b732d-bd69-4c53-8cbe-4361a2bf69b2" />
</td>
<td width="40%" valign="top">

<table>
<tr><th>Hyperparameter</th><th>Value</th></tr>
<tr><td>Optimizer</td><td>Adam</td></tr>
<tr><td>Learning rate</td><td>2e-5</td></tr>
<tr><td>Loss</td><td><code>sparse_categorical_crossentropy</code></td></tr>
<tr><td>Metric</td><td>accuracy</td></tr>
<tr><td>Epochs</td><td>50</td></tr>
<tr><td>Batch size</td><td>20</td></tr>
<tr><td>Trainable base</td><td>No (frozen, but trained end-to-end rather than on cached features)</td></tr>
</table>

</td>
</tr>
</table>

Result: 
Training and validation accuracy tracked each other closely, converging around 77–80%, with both still rising slightly at epoch 50. 
The train–validation gap narrowed substantially compared to Model 2a, and loss curves for both sets declined together and converged near 0.60–0.65 — indicating that overfitting was largely resolved at the cost of a slightly lower peak accuracy than 2a (a typical augmentation trade-off: harder training task, better generalisation).

### Model 2c — Fine-Tuning (selective unfreezing)

<table>
<tr>
<td width="60%" valign="top">
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/99d79696-7aeb-4245-871b-83ca42609b5c" />
</td>
<td width="40%" valign="top">

<table>
<tr><th>Hyperparameter</th><th>Value</th></tr>
<tr><td>Learning rate</td><td>2e-6 (~10× lower than feature-extraction phase)</td></tr>
<tr><td>Augmentation</td><td>Same as Model 2b</td></tr>
</table>

</td>
</tr>
</table>


MobileNetV2 has 16 inverted-residual blocks. Rather than fine-tuning the entire network (risking catastrophic forgetting of useful low-level ImageNet features), only the top blocks were unfrozen:
<img width="1095" height="93" alt="finetune_layers" src="https://github.com/user-attachments/assets/89857cf1-3044-44cd-8b97-08743103ff03" />


Layer range	State	Rationale
Input stem → Block 12	Frozen	Low-level features (edges, textures, colours) — generic across image domains, not food-specific  
Block 13 → Block 16 + `Conv_1`	Unfrozen	High-level, more abstract features — worth adapting specifically to food imagery
```python
conv_base.trainable = True
freeze = True
for layer in conv_base.layers:
    if layer.name == 'block_13_expand':
        freeze = False
    layer.trainable = not freeze
```
Result: Validation accuracy held essentially flat around 0.80–0.804 across all 30 epochs, with training accuracy actually below validation accuracy for most of the run. This is expected behaviour, not a bug: augmentation makes the training set harder than the (unaugmented) validation set, and Dropout is active only during training. There was no evidence of overfitting, but the model appeared to have plateaued / slightly underfit — the learning rate was likely too conservative to make further progress in 30 epochs.  

Fine-tuning added computational cost in terms of longer training time without meaningful gains. The lesson learnt here is that fine-tuning may not be always worth the effort with such negligible benefits. Looking back, I think Model 2b was already performing near the optimal level for my dataset and architecture, therefore I will select Model 2b as the best model.

Hyperparameter tuning summary  

<table>
<tr><th>Model</th><th>LR</th><th>Epochs</th><th>Augmentation</th></tr>
<tr><td>2a</td><td><code>2e-5</code></td><td>30</td><td>No</td></tr>
<tr><td>2b</td><td><code>2e-5</code></td><td>50</td><td>Yes</td></tr>
<tr><td>2c</td><td><code>2e-6</code></td><td>30</td><td>Yes</td></tr>
</table>

# 6. Model Evaluation on the Test Set  
<table>
<tr>
<td width="50%" valign="top">
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/6184f209-4619-4972-8e69-16cff26649bd" />
</td>
<td width="50%" valign="top">
<img width="450" height="300" alt="image" src="https://github.com/user-attachments/assets/8c7c0bb8-aac0-4437-8bb8-02e4e6213bb2" />
</td>
</tr>
</table>

Model 1e achieved a test accuracy of 55.20% with a test loss of 1.7575. In contrast, Model 2b achieved a much higher test accuracy of 82.41% with a substantially lower test loss of 0.5149. These results clearly indicate that Model 2b generalised far better to unseen test images compared to Model 1e.

Model 1's lower accuracy and higher loss tell me it struggled to learn meaningful features from my limited food dataset. Even with data augmentation, training a CNN from scratch requires massive amounts of labeled data. The dataset just wasn't big enough for the model to learn robust representations. The high loss value also shows the model was uncertain about many of its predictions.
Model 2b used a pre-trained MobileNetV2 base with data augmentation added, it already knew how to detect edges, textures, colors, and shapes from ImageNet's 1.2 million images. These generic features transfer really well to food classification. The result was much higher accuracy and lower loss — meaning the model was both more accurate and more confident in its predictions.  
## Recommendation   
I recommend Model 2b as my final model. Its 82.41% test accuracy is far better than Model 1's 55.20%, and its lower test loss (0.51 vs 1.76) proves it makes more reliable predictions.
This experiment taught me that transfer learning is far more effective than training from scratch when working with limited data

Overall, the evaluation results demonstrate that transfer learning with a pre-trained CNN is a more efficient and accurate approach for this food classification problem. The combination of pre-trained feature extraction, data augmentation, and selective fine-tuning enabled Model 2 to achieve better generalisation performance and produce more accurate predictions on unseen test images.  

# Applying model on real food images

<img width="800" height="390" alt="image" src="https://github.com/user-attachments/assets/846efab0-5721-4ab4-a572-dee8f3813f7e" />

I tested the model on several food images downloaded from the internet, including panna cotta and other dishes. For each image, the model returned a probability distribution across all food classes and selected the highest probability class as its prediction. In a classification model, the softmax layer converts the model’s output into probabilities for each class. All probabilities add up to 1 (or 100%).  

The model correctly identified all the food images, showing that my transfer learning approach generalized well beyond the training and test datasets.
The prediction results showed that the model was generally capable of identifying food images correctly, indicating that the transfer learning approach successfully generalized to unseen images.
Overall, the prediction results demonstrate that the trained CNN model can effectively perform real-life food image classification with good practical performance and reasonable prediction confidence.





