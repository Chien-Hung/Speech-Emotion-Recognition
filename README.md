# Speech-Emotion-Recognition

Pytorch implementation of "3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition".

I follow the [original tensorflow code](https://github.com/xuanjihe/speech-emotion-recognition) and change the tensorflow parts to pytorch ones. Please reference the original github for more details.


## Dependency:

* pytorch
* python_speech_features
* wave
* pickle
* sklearn

## Demo

After download the **IEMOCAP** dataset:

```
python zscore.py
python ExtractMel.py
python model.py
```
---

or you can download the processed file, [IEMOCAP.pkl](https://drive.google.com/file/d/18bYKQEjrWB8opvV-eEIuncAwOCLvTG8v/view?usp=sharing)

```
python model.py
```
---

The best valid_UA of this code is about **0.6619**.

## Reference

https://github.com/xuanjihe/speech-emotion-recognition

