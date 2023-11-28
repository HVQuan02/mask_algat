# Masked Feature Modelling: Feature Masking for the unsupervised pre-training of a Graph Attention Network block for bottom-up video event recognition

## Introduction
In this repository we provide the materials for Masked Feature Modelling, a novel approach for the unsupervised pre-training of a Graph Attention Network (GAT) block.
MFM utilizes a pretrained Visual Tokenizer to reconstruct masked features of objects within a video, leveraging the MiniKinetics dataset. We then incorporate the pre-trained GAT block into a state-of-the-art bottom-up supervised video-event recognition architecture, ViGAT [1], to improve the model's starting point and overall accuracy. Experimental evaluations on the YLI-MED dataset [2] demonstrate the effectiveness of MFM in improving event recognition performance.

![unsupervised](https://github.com/bmezaris/masked-ViGAT/assets/33573818/233e55b2-008d-470b-a13c-959febffefd6)

### Video preprocessing

Before training the model on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading.
Specifically, we sample 25 frames per video for YLIMED [2] and 30 frames per video for MiniKinetics [3].
A global frame feature representation is obtained using a  Vision Transformer (ViT) [4] or CLIP [5] backbone. Moreover, a local frame feature representation is also obtained, i.e., Detic is used as object detector (OD) [6] and a feature representation for each object is obtained by applying the network backbone (ViT or CLIP) for each object region.
After the video preprocessing stage (i.e. running the Detic and network backbone), the dataset root directory must contain the following subdirectories:
* When ViT backbone is used:
  * ```vit_global/```: Numpy arrays of size 9x768 (or 30x768) containing the global frame feature vectors for each video (the 9 (30) frames, times the 768-element vector for each frame).
  * ```vit_local/```: Numpy arrays of size 9x50x768 (or 30x50x768) containing the appearance feature vectors of the detected frame objects for each video (the 9 (30) frames, times the 50 most-prominent objects identified by the object detector, times a 768-element vector for each object bounding box).
* When ClIP backbone is used:
  * ```clip_global/```: Numpy arrays of size 25x1024 (or 30x1024) containing the global frame feature vectors for each video (the 25 (30) frames, times the 2048-element vector for each frame).
  * ```clip_local/```: Numpy arrays of size 25x50x1024 (or 30x50x1024) containing the appearance feature vectors of the detected frame objects for each video (the 25 (30) frames, times the 50 most-prominent objects identified by the object detector, times a 1024-element vector for each object bounding box).

Afterwards,we need to generate tokens for the objects we've extracted. This is done through the tokenizer provided in Beit2 [7] repository.
These tokens, along with the features we've obtained, are used to unsupervisingly train a graph in MiniKinetics. 
After the token creation stage, the dataset root directory must contain the following subdirectories:
* ```tokens/```: Numpy arrays of size 30x50x8192 containing the tokens that are present for each detected object for each frame in a video.

![supervised](https://github.com/bmezaris/masked-ViGAT/assets/33573818/425a88b1-d3d8-4092-8320-e1d7234233d5)

### Training

Initially, using train_masked.py, we generate the new graph, to replace one or more supervised-learned graphs in the ViGAT method.
```
python train_masked.py --dataset_root <dataset dir> --dataset [<minikinetics>]
```
By default, the model weights are saved in the ```weights/``` directory. 

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train_masked.py --help```.

After creating this new graph, we use train_masked_total.py, selecting the appropriate model and parameters and run 
```
python train_masked_total.py --dataset_root <dataset dir> --dataset [<ylimed>]
```
By default, the model weights are saved in the ```weights/``` directory. 

### Evaluation

To evaluate a model, run

```
python evaluate_masked_total.py weights/<model name>.pt --dataset_root <dataset dir> --dataset [<ylimed|>]
```
Again, the evaluation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python evaluate_masked_total.py --help```.

## License

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [2,3], etc.). For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation

If you find our approach useful in your work, please cite the following publication where this approach was proposed:

D. Daskalakis, N. Gkalelis, V. Mezaris, "Masked Feature Modelling for the unsupervised pre-training of a Graph Attention Network block for bottom-up video event recognition", Proc. 25th IEEE Int. Symp. on Multimedia (ISM 2023), Laguna Hills, CA, USA, Dec. 2023.

BibTex:
```
@inproceedings{Daskalakis_ISM2023,
author={Daskalakis, Dimitrios and Gkalelis, Nikolaos and Mezaris, Vasileios},
title={Masked Feature Modelling for the unsupervised pre-training of a Graph Attention Network block for bottom-up video event recognition},
year={2023},
month={Dec.},
booktitle={25th IEEE Int. Symp. on Multimedia (ISM 2023)}
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement 101021866 (CRiTERIA).

## References

[1] N. Gkalelis, D. Daskalakis, V. Mezaris, "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network", IEEE Access, vol. 10, pp. 108797-108816, 2022.

[2] Bernd, Julia, et al. "The YLI-MED corpus: Characteristics, procedures, and plans." arXiv preprint arXiv:1503.04250 (2015).

[3] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy. Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. In Proc. ECCV, 2018, pp. 305-321

[4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. ICLR, Virtual Event, Austria, May 2021.

[5] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

[6] Zhou, Xingyi, et al. "Detecting twenty-thousand classes using image-level supervision." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[6] Peng, Zhiliang, et al. "BEiT v2: Masked image modeling with vector-quantized visual tokenizers." arXiv preprint arXiv:2208.06366 (2022).
