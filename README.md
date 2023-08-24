# Masked Feature Modelling: Feature Masking for the unsupervised pre-training of a Graph Attention Network block for bottom-up video event recognition

## Introduction
In this repository we provide the materials for Masked Feature Modelling, a novel approach for the unsupervised pre-training of a Graph Attention Network (GAT) block.
MFM utilizes a pretrained Visual Tokenizer to reconstruct masked features of objects within a video, leveraging the MiniKinetics dataset. We then incorporate the pre-trained GAT block into a state-of-the-art bottom-up supervised video-event recognition architecture, ViGAT, to improve the model's starting point and overall accuracy. Experimental evaluations on the YLI-MED dataset demonstrate the effectiveness of MFM in improving event recognition performance.

NEW IMAGE

### Video preprocessing

Before training the model on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading.
Specifically, we sample 25 frames per video for YLIMED and 30 frames per video for MiniKinetics.
A global frame feature representation is obtained using a  Vision Transformer (ViT) [1] or CLIP [2] backbone. Moreover, a local frame feature representation is also obtained, i.e., Detic is used as object detector (OD) [3] and a feature representation for each object is obtained by applying the network backbone (ViT or CLIP) for each object region.
After the video preprocessing stage (i.e. running the Detic and network backbone), the dataset root directory must contain the following subdirectories:
* When ViT backbone is used:
  * ```vit_global/```: Numpy arrays of size 9x768 (or 30x768) containing the global frame feature vectors for each video (the 9 (30) frames, times the 768-element vector for each frame).
  * ```vit_local/```: Numpy arrays of size 9x50x768 (or 30x50x768) containing the appearance feature vectors of the detected frame objects for each video (the 9 (30) frames, times the 50 most-prominent objects identified by the object detector, times a 768-element vector for each object bounding box).
* When ClIP backbone is used:
  * ```clip_global/```: Numpy arrays of size 25x1024 (or 30x1024) containing the global frame feature vectors for each video (the 25 (30) frames, times the 2048-element vector for each frame).
  * ```clip_local/```: Numpy arrays of size 25x50x1024 (or 30x50x1024) containing the appearance feature vectors of the detected frame objects for each video (the 25 (30) frames, times the 50 most-prominent objects identified by the object detector, times a 1024-element vector for each object bounding box).

Afterwards,we need to generate tokens for the objects we've extracted. This is done through the tokenizer provided in Beit2 [4] repository.
These tokens, along with the features we've obtained, are used to unsupervisingly train a graph in MiniKinetics. 
After the token creation stage, the dataset root directory must contain the following subdirectories:
* ```tokens/```: Numpy arrays of size 30x50x8192 containing the tokens that are present for each detected frame object for each video.

### Training

Initially, using train_masked.py, we generate the new graph, to replace one or more supervised-learned graphs in the vigat method.
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

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [1,2,3], etc.). Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement 101021866 (CRiTERIA);

## References

