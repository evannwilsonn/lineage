# Data

Nothing in this folder is committed except this file. [`scripts/get-data`](../scripts/get-data) fills it:

| Path | What | Size |
|---|---|---|
| `splits/train.txt`, `splits/test.txt` | the official Stanford Dogs split (12,000 / 8,580 file names), from `lists.tar`'s `train_list.mat` and `test_list.mat` | 0.8 MB |
| `images/<synset>-<breed>/<file>.jpg` | the 20,580 photos, shrunk by `scripts/prepare_images.py` so the shorter side is at most 272 px (Pillow bicubic, JPEG quality 90) | 650 MB |
| `embeddings/{crop,squash,crop-flip,squash-flip}.f32` | written by `lineage embed`: one row of 1,280 little-endian float32 per photo, training list first, then test list | 4 x 105 MB |

**Source:** http://vision.stanford.edu/aditya86/ImageNetDogs/ (`images.tar`, 757 MB, streamed and never stored; `lists.tar`).
Labels are the folder names; sorted, they are in ImageNet synset order, which is also the order of the 120 outputs.

**Terms:** Stanford Dogs is built from ImageNet images and annotation, and is for non-commercial research and educational
use under ImageNet's terms. Cite: A. Khosla, N. Jayadevaprakash, B. Yao and L. Fei-Fei, "Novel dataset for Fine-Grained
Image Categorization", First Workshop on Fine-Grained Visual Categorization (FGVC), CVPR 2011.

The bounding-box annotations (`annotation.tar`) are not used: the photos are classified whole, as the course does.
