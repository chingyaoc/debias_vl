# Debiasing Vision-Language Models via Biased Prompts

<p align='center'>
<img src='https://github.com/chingyaoc/debias_vl/blob/main/fig.png?raw=true' width='450'/>
</p>


Machine learning models have been shown to inherit biases from their training datasets, which can be particularly problematic for vision-language foundation models trained on uncurated datasets scraped from the internet. The biases can be amplified and propagated to downstream applications like zero-shot classifiers and text-to-image generative models. In this study, we propose a general approach for debiasing vision-language foundation models by projecting out biased directions in the text embedding. In particular, we show that debiasing only the text embedding with a calibrated projection matrix suffices to yield robust classifiers and fair generative models. The closed-form solution enables easy integration into large-scale pipelines, and empirical results demonstrate that our approach effectively reduces social bias and spurious correlation in both discriminative and generative vision-language models without the need for additional data or training.


**Debiasing Vision-Language Models via Biased Prompts**, Preprint 2023 [[paper]](https://arxiv.org/abs/2302.00070)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Varun Jampani](https://varunjampani.github.io/), 
[Yuanzhen Li](https://people.csail.mit.edu/yzli/),
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.6
- PyTorch 1.10.1
- PIL
- diffuser
- scikit-learn
- clip
- transformers

## Code
Check the ```discriminative``` and ```generative``` folders.

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2023debiasing,
  title={Debiasing Vision-Language Models via Biased Prompts},
  author={Chuang, Ching-Yao and Varun, Jampani and Li, Yuanzhen and Torralba, Antonio and Jegelka, Stefanie},
  journal={arXiv preprint 2302.00070},
  year={2023}
}
```

For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).

## Acknowledgements
The code of discriminative model is primarily inspired by the supplement of [Zhang and Ré](https://openreview.net/forum?id=uPdS_7pdA9p).

The code of generative model is primarily inspired by the huggingface [example](https://github.com/huggingface/diffusers/tree/main/examples).


