## GCMSFormer

This is the code repo for GCMSFormer mehtod. We proposed the GCMSFormer for resolving the overlapped peaks in complex 
GC-MS data based on a Transformer model. The GCMSFormer model was trained, validated, and tested with 30,000 augmented 
simulated overlapped peaks in a ratio of 8:1:1, and its bilingual evaluation understudy (BLEU) on the test set was 0.98774. 
With the aid of the orthogonal projection resolution method (OPR), GCMSFormer can predict the pure mass spectra of all 
components in overlapped peaks (mass spectral matrix S), and then use the least squares method to find the concentration 
distribution matrix C. The automatic resolution of the overlapped peaks can be easily achieved.

![](https://github.com/Xuaner111/GCMSFormer/blob/master/workflow.png)

### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)
- [pytorch](https://pytorch.org/) 

By using the [`environment.yml`](https://github.com/Xuaner111/GCMSFormer/blob/master/environment.yml), [`requirements.txt`](https://github.com/Xuaner111/GCMSFormer/blob/master/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/Xuaner111/GCMSFormer.git
    cd GCMSFormer
    conda env create -f environment.yml
    conda activate GCMSFormer
    
## Data augmentation

The overlapped peak dataset for training, validating and testing the GCMSFormer model is obtained using the [gen_datasets](https://github.com/Xuaner111/GCMSFormer/blob/master/GCMSFormer/da.py#L248) functions.

    TRAIN, VALID, TEST, tgt_vacob = gen_datasets(para)

*Optionnal args*
- para : Data augmentation parameters 

## Model training
Train the model based on your own training dataset with [train_model](https://github.com/Xuaner111/GCMSFormer/blob/master/GCMSFormer/GCMSformer.py#L428) function.

    model, Loss = train_model(para, TRAIN, VALID, tgt_vacob)

*Optionnal args*
- para : Hyperparameters for model training
- TRAIN : Training set
- VALID : Validation set
- tgt_vacob : Library

## Resolution

Automatic Resolution of GC-MS data files by using the [Resolution](https://github.com/Xuaner111/GCMSFormer/blob/master/GCMSFormer/Resolution.py#L49) function.

    Resolution(path, filename, model, tgt_vacob, device)
    
*Optionnal args*
- path : GC-MS data path
- filename : GC-MS data filename
- model : GCMSFormer model
- tgt_vacob : Library
- device : Data distribution devices (cuda/cpu)

## Clone the repository and run it directly
[git clone](https://github.com/Xuaner111/GCMSFormer)

An example has been provided in [test_new.ipynb](https://github.com/Xuaner111/GCMSFormer/blob/master/test_new.ipynb) 
script for the convenience of users. The GC-MS file used in it is available in the file [data](https://github.com/Xuaner111/GCMSFormer/tree/master/data).

## Contact
- 549648385@qq.com
