# Food Image Classification with MobileNetV2 (Transfer Learning)  
Author: Ng Chee Wee (3303451Y)

Objective: Classify images of food into 10 categories using transfer learning with a pre-trained MobileNetV2 convolutional base.  

This README documents the full pipeline implemented in `DLIR_assignment_model_2_NgCheeWee.ipynb` — data loading, preprocessing, model architecture, hyperparameter tuning across three iterations, evaluation results, and a final recommendation.  

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

Layer	Setting	Effect

`RandomFlip`	`"horizontal"`	Random left-right mirroring

`RandomRotation`	±10% of 360°	Random small rotations

`RandomZoom`	±20% (height & width)	Random zoom in/out

`RandomTranslation`	±20% (height & width)	Random shifting

Augmentation policy by split:

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

`GlobalAveragePooling2D` over `Flatten` — collapses the 7×7×1280 feature map to a 1280-length vector instead of a much larger flattened vector. This keeps the number of parameters in the following Dense layer low, reduces overfitting risk, and is the standard choice for transfer learning on modern architectures.

Dropout (0.3) before the output layer acts as a regulariser against overfitting once training moved beyond the simple feature-extraction baseline.
Softmax output over 10 units, paired with `sparse_categorical_crossentropy`, matches the integer-encoded labels produced by `image_dataset_from_directory`.

---
# 5. Hyperparameters & Training Iterations

### Model 2a — Feature Extraction (frozen base, no augmentation)

Hyperparameter	Value

Optimizer	Adam  
Learning rate	2e-5  
Loss	`sparse_categorical_crossentropy`  
Metric	accuracy  
Epochs	30  
Batch size	20  
Trainable base	No (fully frozen)  

Result: 
Training accuracy climbed to ~84%; validation accuracy plateaued around 80–81% by epoch 30. Validation loss flattened near 0.58 while training loss kept falling — a sign of mild overfitting, as the gap between training and validation accuracy widened over time.

### Model 2b — Feature Extraction + Data Augmentation (frozen base)

Hyperparameter	Value  
Optimizer	Adam  
Learning rate	2e-5  
Loss	`sparse_categorical_crossentropy`  
Metric	accuracy  
Epochs	50  
Batch size	20  

Trainable base	No (frozen, but trained end-to-end rather than on cached features)

Augmentation	Flip, rotation, zoom, translation (train set only)

Result: 
Training and validation accuracy tracked each other closely, converging around 77–80%, with both still rising slightly at epoch 50. 
The train–validation gap narrowed substantially compared to Model 2a, and loss curves for both sets declined together and converged near 0.60–0.65 — indicating that overfitting was largely resolved at the cost of a slightly lower peak accuracy than 2a (a typical augmentation trade-off: harder training task, better generalisation).

### Model 2c — Fine-Tuning (selective unfreezing)

Hyperparameter value  
Learning rate	2e-6 (~10× lower than feature-extraction phase)  
Augmentation	Same as Model 2b

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

Fine-tuning run 2 (self-experiment, learning-rate tuning):
To address the plateau, the learning rate was raised 5× from 2e-6 to 1e-5:
Hyperparameter	Value
Optimizer	Adam
Learning rate	1e-5
Epochs	30
Augmentation	Same as Model 2b
Result: This was the best-performing configuration. Validation accuracy improved to ~0.809, training accuracy converged towards validation accuracy (closing the earlier gap), and both training and validation loss trended consistently downward across all 30 epochs — the healthiest-looking curves of all three models.
Hyperparameter tuning summary
Model	LR	Epochs	Augmentation	Base trainable	Best val. accuracy
2a	2e-5	30	No	Frozen	~0.806
2b	2e-5	50	Yes	Frozen	~0.801
2c (run 1)	2e-6	30	Yes	Blocks 13–16 + Conv_1	~0.804 (plateau)
2c (run 2)	1e-5	30	Yes	Blocks 13–16 + Conv_1	~0.809 (best)
Key tuning insight: the 10× learning-rate-reduction rule of thumb for fine-tuning (2e-6) was too conservative for this dataset/architecture combination — it stalled progress. Raising it to 1e-5 (5× lower than the feature-extraction LR, rather than 10×) unlocked further gains without destabilising training or causing the loss to diverge.
---
6. Model Evaluation on the Test Set
This is the one part of the notebook that needs to be read carefully, because the test-set evaluation results are inconsistent across cells and should not all be taken at face value:
Cell	Test set used	Reported test accuracy	Reliability
Model 2a (`model_fe.evaluate(test_features, test_labels)`)	Pre-computed features from the true `test/` directory	0.7780	✅ Reliable — correct, untouched test set
Model 2b, first evaluation attempt	`train_dataset` mapped with `data_preprocess` and mistakenly re-assigned to the variable `test_dataset` (a copy-paste/variable-naming bug — this is actually a subset of the training data, evaluated with `steps=50`)	0.8420	❌ Not a valid test result — measuring training data, not held-out test data
Model 2b, second evaluation (`model2.evaluate(test_dataset)` immediately after)	Same mislabelled `test_dataset` variable (still training data)	0.8241	❌ Same issue — inflated, not a true test accuracy
Final `## Evaluate` section, loading `food_model_fe_aug.keras` fresh	By this point `test_dataset` had been overwritten again and no longer points at the original `test_dir` images	0.1060	⚠️ Near-random (1/10 classes) — almost certainly evaluating against mismatched/incompatible data (e.g. raw un-preprocessed images, or a dataset object pointing at the wrong directory/pipeline)
Root cause: the variable name `test_dataset` was reused and overwritten multiple times throughout the notebook — at one point being reassigned to a mapped version of `train_dataset` instead of the original `test_dir`-derived dataset. By the final evaluation cell, this had cascaded into evaluating the saved model against data that no longer matches what the model expects, producing a near-random ~10.6% accuracy that should not be reported as the model's true performance.
The only fully trustworthy, leak-free test accuracy in the notebook is Model 2a's 0.7780, since `test_features`/`test_labels` were derived directly and only once from the untouched `test/` directory before any variable got overwritten.
Recommended fix for re-running evaluation cleanly
```python
test_dataset_clean = image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_dataset_clean = test_dataset_clean.map(data_preprocess, num_parallel_calls=8)

test_loss, test_acc = model.evaluate(test_dataset_clean)
print(f"Test Accuracy: {test_acc:.4f}")
```
Using a dedicated, clearly-named variable (e.g. `test_dataset_clean`) for the final test set — built once and never reused for any other purpose — would have avoided this issue entirely.
---
7. Performance Analysis Summary
Aspect	Model 2a	Model 2b	Model 2c (best run)
Overfitting	Mild (train/val gap widens)	Resolved	Resolved
Underfitting	No	Slight	Slight, but improving
Val. accuracy trend	Plateaus ~80%	Still rising slowly at epoch 50	Improves steadily to ~80.9%
Loss curve health	Train↓, Val flattens	Both ↓, converge well	Both ↓, healthiest of the three
Trustworthy test accuracy	0.778 (only reliable test result)	Not obtainable from notebook (bug)	Not separately evaluated in notebook
Across validation curves, fine-tuning with LR = 1e-5 (Model 2c, run 2) gave the best and most stable validation accuracy (~80.9%) with the cleanest loss curves, narrowly ahead of the frozen-base baseline.
---
8. Final Recommendation
Best model: Model 2c, fine-tuned with learning rate 1e-5. It achieved the highest validation accuracy (~0.809), the smallest train–validation gap, and the most consistent downward-trending loss curves of all three variants — evidence of good generalisation rather than memorisation.
Before deploying or reporting final numbers, re-run test-set evaluation using a freshly-created, clearly-named test dataset object (see Section 6) against the fine-tuned model (`food_model_fine_tuned.keras`). The notebook currently only has a trustworthy test score for Model 2a (0.778); the fine-tuned model's true test accuracy still needs to be measured cleanly.
Data augmentation is worth keeping — it closed the overfitting gap seen in Model 2a with only a small cost to peak training accuracy, and is standard practice for a dataset of this size (7,500 training images for 10 classes).
Selective fine-tuning (blocks 13–16 + `Conv_1` only) is preferable to fine-tuning the entire base — it adapts the high-level, task-specific features to food imagery while preserving the generic low-level edge/texture detectors learned from ImageNet, and trains faster than full unfreezing.
Suggested next steps for further improvement:
Fix the `test_dataset` variable-naming bug and re-evaluate all three saved models (`food_model_fe.keras`, `food_model_fe_aug.keras`, `food_model_fine_tuned.keras`) on a clean, correctly-built test set for a fair side-by-side comparison.
Try a learning-rate schedule (e.g. `ReduceLROnPlateau`) during fine-tuning instead of a single fixed rate, since the 2e-6 run plateaued and only improved after a manual rate increase.
Add a confusion matrix and per-class precision/recall on the test set to identify which of the 10 food categories are most often confused — overall accuracy alone hides class-level weaknesses.
---
9. Repository Structure
```
.
├── DLIR_assignment_model_2_NgCheeWee.ipynb   # Full training/evaluation notebook
├── README.md                                  # This file
└── images/
    ├── model_architecture.png                 # Classifier head + MobileNetV2 base diagram
    └── finetune_layers.png                    # Frozen vs. unfrozen layer split for fine-tuning
```
Saved model checkpoints (produced by the notebook, not included in this repo by default — add via Git LFS if needed):
`food_model_fe.keras` — Model 2a (feature extraction, no augmentation)
`food_model_fe_aug.keras` — Model 2b (feature extraction + augmentation)
`food_model_fine_tuned.keras` — Model 2c (fine-tuned, LR = 1e-5 run)
---
10. Environment
Component	Detail
Framework	TensorFlow / Keras
Pre-trained model	MobileNetV2 (ImageNet weights)
GPU	NVIDIA GeForce RTX 4050 Laptop GPU (CUDA, XLA-compiled ops)
Dataset	10-class food image dataset (7,500 train / 2,000 validation / 500 test images)
