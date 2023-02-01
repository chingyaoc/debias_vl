## Debiasing Text-to-Image Models.
The code aims to remove the gender bias of [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4).

Flags:
  - `--cls`: select the target class, e.g., doctor.
  - `--lr`: learning rate of the iterative optimization
  - `--lam`: hyperparameter lambda of debiasing algorithm


For instance, to reproduce the experiments, run
```
python main.py --cls doctor --lr 0.1 --lam 100
```


## Acknowledgements
The code is primarily inspired by the huggingface [example](https://github.com/huggingface/diffusers/tree/main/examples).
