from config import DATASETS


for dataset in DATASETS:
    DATASETS[dataset]().data_processing()