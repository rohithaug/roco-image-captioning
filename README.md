# Enhanced Medical Image Captioning on ROCO Multimodal dataset using Step-by-Step Distillation

---

## Description:

The aim of this project is to enhance Image Captioning on ROCO Multimodal dataset using step-by-step knowledge distillation. The [ROCO dataset](https://www.kaggle.com/datasets/virajbagal/roco-dataset) contains radiology and non-radiology images, along with textual data such as _semtypes_, _CUI's_. A subset of this dataset is used as development data for the Concept Detection and Caption Prediction Task at [ImageCLEF 2019](https://www.imageclef.org/2019). In this project, we explore the usage of **Vision Image Transformer, GPT-2 Decoder, Med-Alpaca LLM, Langchain, T5, and Visual-Language T5** as part of experimentation to arrive at building a multi-modal framework that predicts captions for medical images.

## Files

1. [configs.py](./configs.py) - Contains the configs for all the scripts in this repository.
2. [t5_version1.py](./t5_version1.py) - Utilizes T5 to predict caption using _semtypes_ only.
3. [gpt2_image_captioning_without_knowledge.py](./gpt2_image_captioning_without_knowledge.py) - Uses Google's Vision Image Transformer to predict caption using radiology images.
4. [gpt2_image_captioning_with_knowledge.py](./gpt2_image_captioning_with_knowledge.py) - Uses Google's Vision Image Transformer to predict caption using radiology images and Med-Alpaca 13B paramater LLM that uses Langchain's Chain of Thought (CoT) API to develop relationships.

## Dependencies

Install the dependencies for this repository by executing the following command.

```
    $ pip install -r requirements.txt
```

## Collaborators

1. [Rohith Sathiamoorthy Pandian](https://github.com/rohithaug) - [https://www.linkedin.com/in/rohithsp/](https://www.linkedin.com/in/rohithsp/)

2. [Rishivardhan Krishnamoorthy](https://github.com/rishivar) - [https://www.linkedin.com/in/rishi-vardhan/](https://www.linkedin.com/in/rishi-vardhan/)

## References

> Bagal, V. (2020). Roco-dataset. https://www.kaggle.com/datasets/virajbagal/roco-dataset.
Last Accessed: 2023-05-10.

> O. Pelka, S. Koitka, J. Rückert, F. Nensa, C.M. Friedrich,  
> ["__Radiology Objects in COntext (ROCO): A Multimodal Image Dataset__"](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04.pdf).  
> MICCAI Workshop on Large-scale Annotation of Biomedical Data and Expert Label Synthesis (LABELS) 2018, September 16, 2018, Granada, Spain. Lecture Notes on Computer Science (LNCS), vol. 11043, pp. 180-189, Springer Cham, 2018.  
> doi: [10.1007/978-3-030-01364-6_20](https://doi.org/10.1007/978-3-030-01364-6_20)

> Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, & Peter J. Liu. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.

> Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, & Neil Houlsby. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners.

> Jaemin Cho, Jie Lei, Hao Tan, & Mohit Bansal (2021). Unifying Vision-and-Language Tasks via Text Generation. In ICML.