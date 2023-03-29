## GCMSFormer

This is the code repo for GCMSFormer mehtod. We proposed the GCMSFormer for resolving the overlapped peaks in complex GC-MS data based on a Transformer model. The GCMSFormer model was trained, validated, and tested with 30,000 augmented simulated overlapped peaks in a ratio of 8:1:1, and its bilingual evaluation understudy (BLEU) on the test set was 0.985. With the aid of the orthogonal projection resolution method (OPR), GCMSFormer can predict the pure mass spectra of all components in overlapped peaks (mass spectral matrix S), and then use the least squares method to find the concentration distribution matrix C. The automatic resolution of the overlapped peaks can be easily achieved. 

<div align="center">
<img src="https://github.com/zmzhang/GCMSFormer/blob/main/workflow.png" width=809 height=970 />
</div>






### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)
- [pytorch](https://pytorch.org/) 

By using the [`environment.yml`](https://github.com/zmzhang/GCMSFormer/blob/main/environment.yml), [`requirements.txt`](https://github.com/zmzhang/GCMSFormer/blob/main/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/zmzhang/GCMSFormer.git
    cd GCMSFormer
    conda env create -f environment.yml
    conda activate GCMSFormer
    
## Data augmentation

The overlapped peak dataset for training, validating and testing the GCMSFormer model is obtained using the [gen_datasets](https://github.com/zmzhang/GCMSFormer/blob/main/GCMSFormer/da.py#L426) functions.

    TRAIN, VALID, TEST, tgt_vacob = gen_datasets(para)

*Optionnal args*
- para : Data augmentation parameters 

## Model training
Train the model based on your own training dataset with [train_model](https://github.com/zmzhang/GCMSFormer/blob/main/GCMSFormer/GCMSFormer.py#L357) function.

    model, Loss = train_model(para, TRAIN, VALID, tgt_vacob)

*Optionnal args*
- para : Hyperparameters for model training
- TRAIN : Training set
- VALID : Validation set
- tgt_vacob : Library

## Resolution

Automatic Resolution of GC-MS data files by using the [Resolution](https://github.com/zmzhang/GCMSFormer/blob/main/GCMSFormer/Resolution.py#L50) function.

    Resolution(path, filename, model, tgt_vacob, device)
    
*Optionnal args*
- path : GC-MS data path
- filename : GC-MS data filename
- model : pSCNN2 model
- tgt_vacob : Library
- device : Data distribution devices (cuda/cpu)

## Clone the repository and run it directly
[git clone](https://github.com/zmzhang/GCMSFormer)

An example has been provided in [test.ipynb](https://github.com/zmzhang/GCMSFormer/blob/main/test.ipynb) script for the convenience of users. Users can run it directly by placing the unzipped data file and GCMSFormer-1.0 file in the same directory after downloading.

## Contact
- fylmgm@163.com
