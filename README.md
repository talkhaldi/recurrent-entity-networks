# Introduction

This is an update to the implementation of EntNet done by [jimfleming](https://github.com/jimfleming/recurrent-entity-networks) to let it take [Children Book Test](https://research.fb.com/downloads/babi/) as input.

# Recurrent Entity Networks

This repository contains an independent TensorFlow implementation of recurrent entity networks from [Tracking the World State with
Recurrent Entity Networks](https://arxiv.org/abs/1612.03969). This paper introduces the first method to solve all of the bAbI tasks using 10k training examples. The author's original Torch implementation is now available [here](https://github.com/facebook/MemNN/tree/master/EntNet-babi).

<img src="assets/diagram.png" alt="Diagram of recurrent entity network architecture" width="886" height="658">

## Results

Accuracy of EntNet when run on CBT according to the paper. Actual results with this repo will be posted later.

Model |  Named Entities | Common Nouns
--- | --- | --- 
EntNet (general) | 0.484 | 0.540
EntNet (simple) | 0.616 | 0.588

## Setup

1. Download the dataset CBTest.tgz from [The bAbI Project](https://research.fb.com/downloads/babi/) and extract it to a folder called CBT.
2. Run [prep_data.py](entity_networks/prep_data.py) which will convert the datasets into [TFRecords](https://www.tensorflow.org/programmers_guide/reading_data#standard_tensorflow_format).
3. Run `python -m entity_networks.main` adding the required arguments. --help will show them.

## Major Dependencies

- TensorFlow v1.4.0

(For additional dependencies see [requirements.txt](requirements.txt))

