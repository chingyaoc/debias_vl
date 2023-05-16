## Debiasing Text-to-Image Models.
The code aims to remove the gender bias of [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

Flags:
  - `--cls`: select the target class, e.g., doctor.
  - `--lam`: hyperparameter lambda of debiasing algorithm


For instance, to reproduce the experiments, run
```
python main.py --cls doctor --lam 500
```


## Acknowledgements
The code is primarily inspired by the huggingface [example](https://github.com/huggingface/diffusers/tree/main/examples).
