# FER2013 — Facial Expression Recognition

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

[Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)
ესაა სახის ემოციების კლასიფიკაციის კონკურსი (ICML 2013 Workshop), რომელიც იყენებს ინტერნეტიდან შეგროვებულ
სურათებს. მონაცემები შედგება 48×48 ზომის ნაცრისფერი (grayscale) სურათებისგან.

- **სამიზნე ცვლადი**: `emotion` — 7 კლასი (`0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral`).
- **შეფასების მეტრიკა**: კლასიფიკაციის **სიზუსტე (accuracy)**.
- **მონაცემების ზომა**: 28,709 სასწავლო, 3,589 public test, 3,589 private test ნიმუში.
- **გამოწვევები**:
  - **კლასების დისბალანსი** — მაგალითად `Disgust`-ს მხოლოდ ~436 ნიმუში აქვს, `Happy`-ს კი ~7,200.
  - **ლეიბლების გარჩევადობა** — `Fear`/`Sad`/`Neutral` ზოგჯერ ადამიანისთვისაც კი ძნელად გასარჩევია; ეს ქმნის ბუნებრივ ზღვარს (ადამიანის სიზუსტე ≈ 65 ± 5 %).
  - **დაბალი რეზოლუცია** — 48×48 პიქსელი ზღუდავს დეტალებს.

## ჩემი მიდგომა პრობლემის გადასაჭრელად

დავალების მთავარი მიზანია **იტერაციული, კარგად გაანალიზებული კვლევა** — და არა პირდაპირი მაღალი შედეგი.
ამიტომ მუშაობა დავიწყე პატარა იმპლემენტაციით და ეტაპობრივად დავამატე სიღრმე/რეგულარიზაცია, თითოეული
გადაწყვეტილების დასაბუთებით. პროცესი ოთხ ფაზად განვითარდა:

1. **ფაზა 1 — საბაზისო (SimpleCNN)**: მცირე 2-ბლოკიანი CNN, რეგულარიზაციის გარეშე, რათა
   ამეწყო სრული pipeline + W&B logging და თვალსაჩინოდ გამოჩენილიყო **overfitting**-ის სურათი.
2. **ფაზა 2 — ღრმა VGG-style CNN**: BatchNorm + Dropout + augmentation + LR scheduler + early stopping,
   რათა გამქრალიყო ფაზა 1-ის train/val სხვაობა; ablation-ებით გაიზომა თითოეული კომპონენტის წვლილი.
3. **ფაზა 3 — ResNet-18**: residual კავშირები, რათა სიღრმეზე ვარჯიში შესაძლებელი გამხდარიყო; მთავარი
   ablation = residual ჩართ./გამორთ. *იგივე* ღრმა ქსელზე.
4. **ფაზა 4 — MobileNet-style CNN**: depthwise-separable კონვოლუციები — სრულად განსხვავებული,
   ეფექტური building block; კითხვა: რამდენად ახლოს მივა VGG-სთან **~72× ნაკლები პარამეტრით**.

ყველა ექსპერიმენტი დაილოგა **Weights & Biases**-ზე  ლოგიკით : project → group → run.

## რეპოზიტორიის სტრუქტურა

```
MLassignment4/
├── README.md                              # პროექტისა და მუშაობის სრული აღწერა (ძირითადი ახსნა)
└── notebooks/
    ├── 01_baseline_simple_cnn.ipynb       # SimpleCNN baseline + LR sweep + რეგულარიზაციის probe
    ├── 02_vgg_style_cnn.ipynb             # VGG-style CNN + 3 ablation
    ├── 03_resnet_cnn.ipynb                # ResNet-18 + residual/label-smoothing ablation
    └── 04_mobilenet_cnn.ipynb             # MobileNet-style (depthwise-separable) + width ablation
```

### ფაილების განმარტება

- **`01_baseline_simple_cnn.ipynb`** — საბაზისო `SimpleCNN` (2 კონვოლუციური ბლოკი `1→32→64`, FC head).
  მოიცავს: მონაცემთა ჩატვირთვა/ვიზუალიზაცია, Dataset/DataLoader, forward/backward sanity შემოწმებები,
  baseline ვარჯიში, learning-rate sweep `{1e-2, 1e-3, 1e-4}` და dropout+weight-decay probe.
- **`02_vgg_style_cnn.ipynb`** — უფრო ღრმა VGG-style CNN (3 ბლოკი `64→128→256`) BatchNorm/Dropout/augmentation-ით,
  LR scheduler-ითა და early stopping-ით. 3 ablation: augmentation off, regularization off, class-weighted loss.
- **`03_resnet_cnn.ipynb`** — `ResNet-18` განლაგების ქსელი residual ბლოკებით. 2 ablation:
  residual off (იგივე ქსელი skip-კავშირის გარეშე) და label smoothing.
- **`04_mobilenet_cnn.ipynb`** — MobileNet-style `MobileNetMini` depthwise-separable კონვოლუციებით
  (~48.5k პარამეტრი). 1 ablation: `width_mult=0.5` (accuracy/size trade-off).

---

## მონაცემთა დამუშავება (Preprocessing)

### წყაროს არჩევანი და დაყოფა

ვიყენებთ `icml_face_data.csv`-ს, რადგან მის `Usage` სვეტში მოცემულია **ოფიციალური დაყოფა ლეიბლებით
სამივე ნაკრებისთვის** (raw `test.csv` ლეიბლების გარეშეა, ამიტომ რეალური test სიზუსტის გამოთვლა მისით ვერ ხერხდება):

| ნაკრები | `Usage` მნიშვნელობა | ზომა | როლი |
|---|---|---|---|
| Train | `Training` | 28,709 | ვარჯიში |
| Val | `PublicTest` | 3,589 | მოდელის შერჩევა / early stopping |
| Test | `PrivateTest` | 3,589 | საბოლოო შეფასება (კონკურსის private ნაკრები) |

### პიქსელების დამუშავება და ნორმალიზაცია

- **პარსინგი**: `pixels` სტრიქონი (2304 ჰარით გამოყოფილი მნიშვნელობა) გადამყავს `(48, 48)` `uint8` მასივად.
- **`ToTensor()`** — პიქსელებს სვამს `[0, 1]` დიაპაზონში.
- **`Normalize([0.5], [0.5])`** — ცენტრავს `[-1, 1]`-ში, რაც აოპტიმიზირებს პროცესს.

### Data Augmentation

- **საბაზისო (SimpleCNN)**: augmentation-ის **გარეშე** — განზრახ, რათა overfitting მკაფიოდ გამოჩენილიყო.
- **VGG/ResNet**: `RandomHorizontalFlip` + `RandomAffine(degrees=10, translate=(0.1, 0.1))`,
  გამოყენებული **მხოლოდ train loader-ზე** (val/test არასოდეს augment-დება).

### კლასების დისბალანსის მართვა

გამოვთვალე ინვერსიული სიხშირის **class weights** (`Disgust ≈ 9.4`, `Happy ≈ 0.57`) და ცალკე
ექსპერიმენტში გადავეცი `CrossEntropyLoss(weight=...)`-ს, რათა შემემოწმებინა მისი ეფექტი იშვიათ კლასებზე.

---

## Training

### ტესტირებული მოდელები და ექსპერიმენტები

ქვემოთ მოცემულია **ყველა** გაშვებული run. Val/Test = საუკეთესო წონების შედეგი; gap = max(train) − best(val).

| # | მოდელი / Run | Train acc | Val acc | **PrivateTest** | gap | ანალიზი |
|---|---|---|---|---|---|---|
| 1 | **SimpleCNN** baseline (lr 1e-3, 30ep) | 0.990 | 0.549 | 0.529 | **0.477** | **მძიმე Overfit** — train→0.99, val ჩერდება ~0.55-ზე, ხოლო val loss იზრდება 1.2→4.3. FC head იზეპირებს მონაცემებს. |
| 2 | SimpleCNN lr 1e-2 (15ep) | — | — | 0.431 | — | **არასტაბილური / underfit** — მაღალი LR ხელს უშლის კონვერგენციას. |
| 3 | SimpleCNN lr 1e-3 (15ep) | — | — | 0.539 | — | საუკეთესო LR — ბალანსი სიჩქარესა და სტაბილურობას შორის. |
| 4 | SimpleCNN lr 1e-4 (15ep) | — | — | 0.500 | — | **ძალიან ნელი** — 15 ეპოქა არ ჰყოფნის. |
| 5 | SimpleCNN reg (dropout 0.5 + wd 1e-4, 15ep) | — | — | **0.568** | — | **რეგულარიზაცია ეხმარება** — SimpleCNN-შიც კი აუმჯობესებს baseline-ს. ეს ამართლებს NB02-ის მიმართულებას. |
| 6 | **VGG-CNN** full (BN+drop+wd+aug) | 0.743 | **0.688** | **0.696** | **0.054** | **საუკეთესო ბალანსი** — gap ჩამოიშალა 0.47→0.05, val loss აღარ ფეთქდება. **საუკეთესო მოდელი.** |
| 7 | VGG-CNN noaug | 0.846 | 0.664 | 0.645 | ~0.18 | **augmentation მთავარი რეგულარიზატორია** — მის გარეშე −5 ქულა, სწრაფი overfit (train→0.85 ეპოქა 14-ზე). |
| 8 | VGG-CNN noreg (dropout 0, wd 0) | 0.785 | 0.681 | 0.696 | ~0.10 | full-ის ტოლი test, მაგრამ ~2× დიდი gap და მზარდი val loss — dropout/wd ამატებს **მდგრადობას**, არა პიკურ სიზუსტეს. |
| 9 | VGG-CNN classweights | — | — | 0.664 | — | **ვაჭრობაა** — Disgust recall 0.64→0.80, მაგრამ ჯამში −3 ქულა, macro-F1 0.68→0.63. |
| 10 | **ResNet-18** residual (40ep, stop 33) | 0.774 | 0.679 | **0.685** | 0.095 | გლუვი ვარჯიში ეპოქა 1-დან; residual-ი მუშაობს, მაგრამ VGG-ს ვერ ჯობნის. |
| 11 | ResNet-18 no-residual (25ep) | 0.624 | 0.607 | 0.614 | ~0.02 | **degradation problem** — ოპტიმიზაცია იჭედება (~0.27 acc 3 ეპოქა), რთული კლასები იშლება (Disgust recall→0.13). |
| 12 | ResNet-18 label-smoothing 0.1 (25ep) | — | 0.640 | 0.655 | — | მიაღწია ლიმიტს *ისევ მზარდი*. |
| 13 | **MobileNet** full (depthwise-separable, ~48.5k params) | 0.592 | 0.586 | 0.605 | **0.006** | **ყველაზე ეფექტური** — VGG-ის ~87% სიზუსტე ~72× ნაკლები პარამეტრით; gap ≈ 0 (ოდნავ underfit). |
| 14 | MobileNet width 0.5 (~4× პატარა) | — | — | 0.510 | — | accuracy/size trade-off — ჩანელების განახევრება აკლებს 9.5 ქულას. |

### Overfit / Underfit ანალიზი

- **მკვეთრი Overfit**: `SimpleCNN baseline` (gap 0.477; train→0.99, val loss იზრდება). 
  capacity ბევრია, რეგულარიზაცია არ არის, ამიტომ მოდელი იზეპირებს.
- **მკვეთრი Underfit**: `SimpleCNN lr 1e-2` (არ კონვერგირდება) და `ResNet no-residual`
  (skip-კავშირის გარეშე ღრმა ქსელი ვერ ვარჯიშდება — vanishing/degradation).
- **ტევადობით შეზღუდული (capacity-limited)**: `MobileNet` — gap ≈ 0.006, train≈val≈0.59, ანუ არ
  იზეპირებს, უბრალოდ პატარაა; შემზღუდველი მისი ზომაა და არა რეგულარიზაცია.
- **კარგი ბალანსი**: `VGG-CNN full` (gap 0.054) და `ResNet residual` (gap 0.095).

### Forward / Backward შემოწმებები

ყოველ ნოუთბუქში, გრძელ ვარჯიშამდე:
- **Forward შემოწმება** — dummy batch-ი უნდა მისცემდეს ფორმას `(batch, 7)`.
- **Backward შემოწმება** — ერთი batch-ის overfit; loss უნდა დაეცეს ~0-მდე (SimpleCNN ~0.002, ResNet ~0.0001),
  რაც ადასტურებს, რომ მოდელი/loss/optimizer სწორად არის შეერთებული.

### Hyperparameter ოპტიმიზაციის მიდგომა

| Hyperparameter | გატესტილი მნიშვნელობები |
|---|---|
| `learning_rate` | {1e-2, 1e-3, 1e-4} (SimpleCNN sweep) |
| `dropout` | {0, 0.3, 0.4, 0.5} |
| `weight_decay` | {0, 1e-4} |
| `optimizer` | Adam (default), SGD+momentum (დოკუმენტირებული ალტერნატივა) |
| `label_smoothing` | {0, 0.1} |
| `scheduler` | ReduceLROnPlateau (factor 0.5, patience 3, val_loss-ზე) |
| `early stopping` | val_loss-ზე, patience 6–10, საუკეთესო წონების აღდგენით |

LR sweep ცხადად აჩვენებს ე.წ. „ზარის ფორმის" დამოკიდებულებას: 1e-2 ძალიან მაღალია (0.431), 1e-4 ძალიან
დაბალი (0.500), 1e-3 ოპტიმალური — ამიტომ შემდგომ მოდელებში default = 1e-3.

### საბოლოო მოდელის შერჩევის დასაბუთება

**საუკეთესო მოდელი: `VGG-CNN full` — PrivateTest 0.696.**

| არქიტექტურა | პარამეტრები | საუკეთესო val | **PrivateTest** | gap |
|---|---|---|---|---|
| SimpleCNN (NB01) | 1.2M | 0.549 | 0.529 | 0.477 |
| **VGG-CNN (NB02)** | 3.5M | 0.688 | **0.696** ⬅ საუკეთესო შედეგი | 0.054 |
| ResNet-18 (NB03) | 11.2M | 0.679 | 0.685 | 0.095 |
| MobileNet (NB04) | **48.5k** | 0.586 | 0.605 | 0.006 |

დასკვნები:

1. **სიღრმე + რეგულარიზაცია + augmentation** (VGG) აგვარებს overfitting-ის სხვაობას: 0.477 → 0.054
   და test 0.529 → 0.696 (**+17 ქულა**).
2. **residual კავშირები მუშაობს** — ablation-ით +7 ქულა (0.614 → 0.685) იდენტურ ღრმა ქსელზე;
   მათ გარეშე ოპტიმიზაცია ცხადად ფერხდება.
3. **დიდი ≠ უკეთესი** — ResNet-18 (11M) **ვერ ჯობნის** VGG-CNN-ს (3.5M) და ოდნავ მეტად overfit-დება.
   FER2013-ზე VGG-ის მასშტაბის ზემოთ ტევადობა აღარ აუმჯობესებს შედეგს; ლიმიტი არის **მონაცემების/ლეიბლების
   გარჩევადობა**, და არა მოდელის ზომა. ოთხივე ქსელი ჩერდება იმავე Fear/Sad/Neutral ემოციების არევისას.
4. **პატარა ≠ უსარგებლო** — MobileNet (48.5k პარამეტრი, **~72× ნაკლები**) აღწევს 0.605-ს, ანუ VGG-ის
   სიზუსტის **~87%**-ს. ეს ადასტურებს იმავე დასკვნას მეორე მხრიდან: რაკი ზედა ზღვარს მონაცემები აწესებენ,
   ეფექტურ პატარა ქსელსაც მცირე გეფით შეუძლია იქამდე მიღწევა. cost — Disgust-ის recall ეცემა 0.13-მდე
   (ტევადობით შეზღუდული მოდელი ივიწყებს იშვიათ კლასს).

---

## W&B Tracking

### ექსპერიმენტების ბმული

W&B Project: https://wandb.ai/smama23-free-university-of-tbilisi-/fer2013-emotion-recognition

ლოგირების სტრუქტურა:
- **Project**: `fer2013-emotion-recognition`.
- **Group** = არქიტექტურა → ერთი არქიტექტურის ყველა run ერთადაა:
  - `SimpleCNN` — 5 run (`SimpleCNN-baseline-lr0.001-bs64`, `SimpleCNN-lr0.01-bs64`, `SimpleCNN-lr0.001-bs64`, `SimpleCNN-lr0.0001-bs64`, `SimpleCNN-reg-dropout0.5-wd1e-4`).
  - `VGG-CNN` — 4 run (`VGG-CNN-full-lr0.001-bs64`, `VGG-CNN-noaug`, `VGG-CNN-noreg`, `VGG-CNN-classweights`).
  - `ResNet-CNN` — 3 run (`ResNet-residual-lr0.001-bs64`, `ResNet-noresidual`, `ResNet-labelsmooth`).
  - `MobileNet-CNN` — 2 run (`MobileNet-full-lr0.001-bs64`, `MobileNet-width0.5`).
- **Run** = ერთი ჰიპერპარამეტრების კონფიგურაცია.

### ჩაწერილი მეტრიკების აღწერა

თითოეული run-ის შიგნით ფიქსირდება:

**Config (პარამეტრები)**: `arch`, `epochs`, `batch_size`, `lr`, `optimizer`, `weight_decay`,
`dropout`, `augment`, `scheduler`, `patience`, `class_weights`, `label_smoothing` (+ `use_residual`,
`layers`, `channels` ResNet-ში).

**მეტრიკები ეპოქებად** (`wandb.log`):
- `train_loss`, `train_acc`, `val_loss`, `val_acc`
- `lr` — scheduler-ის მიერ შემცირებული learning rate
- `wandb.watch(model, log="all")` — გრადიენტებისა და წონების ჰისტოგრამები (backward-ის შემოწმება)

**Summary (საბოლოო)**:
- `test_acc`, `test_loss` — PrivateTest-ზე საუკეთესო წონებით
- `best_val_acc`, `epochs_run`, `overfit_gap`

**Artifacts / Plots**:
- `confusion_matrix` — `wandb.plot.confusion_matrix` PrivateTest-ზე
- გავარჯიშებული მოდელი **ONNX**-ში ექსპორტირებული და დალოგილი როგორც W&B **Artifact**
  (type `model`, `test_acc` metadata-ით)

### საუკეთესო მოდელის შედეგები

**`VGG-CNN-full-lr0.001-bs64`** (group `VGG-CNN`):

| მეტრიკა | მნიშვნელობა |
|---|---|
| Train acc | 0.743 |
| Best Val acc | 0.688 |
| **PrivateTest acc** | **0.696** |
| Train–val gap | 0.054 |
| Epochs (early-stopped) | 39 / 60 |
| ყველაზე რთული კლასი | Fear (recall 0.47) |
| ყველაზე მარტივი კლასი | Happy (F1 0.89), Surprise (F1 0.78) |