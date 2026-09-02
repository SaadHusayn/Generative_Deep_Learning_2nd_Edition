# Lab 1 — Results Write-Up

**Model (identical for both):** `Flatten → Dense(200, ReLU) → Dense(150, ReLU) → Dense(10, softmax)`
**Training:** Adam (lr=0.0005), categorical cross-entropy, 10 epochs, batch size 32, inputs scaled to [0,1]
**Parameters:** 188,660 (Fashion-MNIST) vs. 646,260 (CIFAR-10)

---

## 1. Final Test Performance

| Dataset | Test Acc | Test Loss | Train Acc | Train Loss |
|---|---|---|---|---|
| Fashion-MNIST | **88.76%** | **0.3300** | 91.62% | 0.2213 |
| CIFAR-10 | **47.87%** | **1.4613** | 51.06% | 1.3738 |

Fashion-MNIST beat CIFAR-10 by **40.9 percentage points** on identical architecture and training budget.

---

## 2. Training Loss per Epoch

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.4935 | 0.3598 | 0.3235 | 0.3005 | 0.2811 | 0.2658 | 0.2525 | 0.2417 | 0.2320 | 0.2213 |
| CIFAR-10 | 1.8461 | 1.6561 | 1.5915 | 1.5358 | 1.4973 | 1.4663 | 1.4400 | 1.4139 | 1.3894 | 1.3738 |

**Observation.** Fashion-MNIST drops steeply in two epochs and flattens near a low floor. CIFAR-10 starts higher and falls only 26% across the whole run, still descending at epoch 10. The key detail: CIFAR-10's *training* accuracy only reaches 51.06%, and both datasets show near-identical train→test gaps (2.9 vs 3.2 pts). So CIFAR-10 is **underfitting**, not overfitting — despite 3.4× more parameters.

---

## 3. Sample Predictions

### Fashion-MNIST — 8/10 correct

| # | True | Predicted | ✓ |
|---|---|---|---|
| 1 | Sneaker | Sneaker | ✓ |
| 2 | Shirt | Pullover | ✗ |
| 3 | Ankle boot | Ankle boot | ✓ |
| 4 | Coat | Coat | ✓ |
| 5 | Trouser | Trouser | ✓ |
| 6 | Trouser | Trouser | ✓ |
| 7 | Pullover | Pullover | ✓ |
| 8 | Dress | Dress | ✓ |
| 9 | T-shirt/top | Dress | ✗ |
| 10 | T-shirt/top | T-shirt/top | ✓ |

### CIFAR-10 — 6/10 correct

| # | True | Predicted | ✓ |
|---|---|---|---|
| 1 | horse | horse | ✓ |
| 2 | cat | truck | ✗ |
| 3 | dog | horse | ✗ |
| 4 | deer | deer | ✓ |
| 5 | deer | horse | ✗ |
| 6 | ship | ship | ✓ |
| 7 | ship | ship | ✓ |
| 8 | deer | frog | ✗ |
| 9 | frog | frog | ✓ |
| 10 | automobile | automobile | ✓ |

**Error pattern.** Both Fashion-MNIST errors sit in the upper-garment cluster (shirt→pullover, T-shirt→dress); every distinctive silhouette was correct. Three of CIFAR-10's four errors are brown-animal-on-natural-background (dog→horse, deer→horse, deer→frog), and both correct ships were on open blue water — evidence the model keys on background colour, not object shape.

---

## 4. Why CIFAR-10 Is Harder

Fashion-MNIST images are 28×28 grayscale, centred and shot on blank backgrounds, so a garment's silhouette lands in nearly the same pixels every time and a Dense layer can exploit that fixed mapping directly. CIFAR-10 gives 3,072 inputs instead of 784, but the extra dimensions are mostly nuisance variation — objects vary in scale, pose and position against cluttered scenes, so the same class rarely activates the same pixels twice. Colour adds channels without much discriminative power, since a brown dog and a brown horse share a palette while two differently-coloured cars do not; our predictions show exactly this. Class similarity compounds it asymmetrically: Fashion-MNIST's confusable classes are concentrated in one cluster, leaving trouser, bag, sandal, sneaker and ankle boot separable on outline alone, whereas CIFAR-10's cat/dog, automobile/truck and deer/horse are all mutually confusable. Underneath it all, `Flatten` destroys spatial locality — cheap on centred grayscale clothing, fatal on CIFAR-10, which is why 3.4× more parameters still underfit.

---

## 5. Takeaway

CIFAR-10 needs an architecture that preserves spatial structure, not more Dense units: convolution with pooling for translation invariance, plus batch norm and augmentation. Fashion-MNIST is already near the MLP ceiling — its remaining error is dominated by the shirt/pullover/coat cluster.
