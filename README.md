# FGNN2
This is the official open source repository for the paper

**[Functionality matters in netlist representation learning](https://dl.acm.org/doi/pdf/10.1145/3489517.3530410)**, ACM/IEEE Design Automation Conference (**DAC**), 2022.

Ziyi Wang, Chen Bai, Zhuolun He, Guangliang Zhang, Qiang Xu, Tsung-Yi Ho, Yu Huang, Bei Yu

Citation:

```
@inproceedings{wang-dac2022-fgnn,
    title={Functionality matters in netlist representation learning},
    author={Wang, Ziyi and Bai, Chen and He, Zhuolun and Zhang, Guangliang and Xu, Qiang and Ho, Tsung-Yi and Yu, Bei and Huang, Yu},
    booktitle=ACM/IEEE Design Automation Conference,
    pages={61--66},
    year={2022}
}
```

and **[FGNN2: a Powerful Pre-training Framework for Learning the Logic Functionality of Circuits](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10609964)**, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (**TCAD**), 2024.

Ziyi Wang, Chen Bai, Zhuolun He, Guangliang Zhang, Qiang Xu, Tsung-Yi Ho, Yu Huang, Bei Yu

Citation:

```
@article{wang-tcad24-fgnn2,
    title={FGNN2: a Powerful Pre-training Framework for Learning the Logic Functionality of Circuits},
    author={Wang, Ziyi and Bai, Chen and He, Zhuolun and Zhang, Guangliang and Xu, Qiang and Ho, Tsung-Yi and Yu, Bei and Huang, Yu},
    journal=IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems,
    year={2024}
}
```


## Setup
The experiments are conducted on Linux, with Python version 3.7.13, PyTorch version 1.13.0, and Dgl version 0.8.1.
 
## Pretraining Dataset
Please refer to [our Pretraining Dataset repo](https://github.com/FGNN2/FGNN2_pretraindata) for download instructions and documentation.

## Pretrain
First, download the pretaining dataset and extract it to `<repo root>/datasets/pretrain_data/`.

You can also choose to generate the pretraining dataset using the rawdata provided. You can first extract the rawdata to `<repo root>/rawdata/pretrain_data/`, go to the folder `<repo root>/src` and run the following command for i = 4...7

```shell
python pretrain_dataparser.py --rawdata_path ../rawdata/pretrain_data --datapath ../datasets/pretrain_data/ --num_input i
```

After you get the pretraining dataset, go to the folder `<repo root>/src` and run the following command:

``` shell
python pretrain.py  --flag_inv --weighted --datapath ../datasets/pretrain_data --checkpoint ../checkpoints/example
```

We have also provided our pretrained weight, which can be downloaded [here](https://drive.google.com/file/d/1NGAJRedx040A4EeN9mALXsly87TRseRF/view?usp=share_link).

## Downstream task

Here we show an example application of our pretrained FGNN2 model on the netlist classification task. Suppose that the pretrained weight is put under `<repo root>/checkpoints/example/weight.pth`.

Firstly, download the global task dataset [here](https://drive.google.com/file/d/1C5ZTyWL2yU9QBV7L3gP4J4zhfiypzoxl/view?usp=share_link) and extract it to `<repo root>/datasets/global_data/`.

go to the folder `<repo root>/src` and run the following command:

``` shell
python train_global.py --datapath ../dataset/global_data/ --batch_size 128 --nlabels 4  --ratio 0.5 --flag_inv --pre_train --start_point `../checkpoints/example/weight.pth`  --checkpoint ../checkpoints/global/example 
```

### Code structure
`FunctionConv.py`: our customized GNN model implementation.

`options.py`: the hyperparameters.

`pretrain.py`: the model pretraining logic.

`pretrain_dataparser.py`: the data reader and pre-processor for pertaining.

`train_local.py`: the local downstream task training/fine-tuning and testing logic.

`train_global.py`: the global downstream task training/fine-tuning and testing logic.

`truthvalue_parser.py`: the parser for truthtable circuits.

