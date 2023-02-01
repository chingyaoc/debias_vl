## Debiasing Zero-Shot Models.

Flags:
  - `--dataset`: select the dataset (waterbirds/celebA)
  - `--load_base_model`: select backbone model (clip_RN50/clip_ViTL14)
  - `--debias`: debias the text embedding or not
  - `--lam`: hyperparameter lambda of debiasing algorithm


For instance, to reproduce the experiments, run
```
python main.py --dataset waterbirds --load_base_model clip_RN50 --debias
```


## Acknowledgements
The code is primarily inspired by the supplement of [Zhang and Ré](https://openreview.net/forum?id=uPdS_7pdA9p).
