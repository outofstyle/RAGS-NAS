Installation
pip install -r requirements.txt


install [NAS-Bench-101](https://github.com/google-research/nasbench) and download nasbench_only108.tfrecord into datasets/NASBench101 
download [NAS-Bench-201-v1_0-e61699.pth](https://github.com/D-X-Y/NAS-Bench-201) into datasets/NASBench201
install [NAS-Bench-301(nasbench301/nasbench301 folder)](https://github.com/crwhite14/nasbench301) and save in datasets/nasbench301. Save NAS-Bench-301 Models in datasets/nasbench301/ Save NAS-Bench-301 Data in datasets/NASBench301/


Usage

Define directory path in Settings.py
run generation/Training_Generator_NB301.py for pretraining the generative model on NAS-Bench-301
then run search_EA/search_EA_NB301.py search on NAS-Bench-301

run generation/Training_Generator.py for pretraining the generative model on NAS-Bench-101 and NAS-Bench-201
then run search_EA/search_EA.py search on NAS-Bench-101 and NAS-Bench-201
Acknowledgement

Code base from [AG-Net](https://github.com/jovitalukasik/AG-Net)
