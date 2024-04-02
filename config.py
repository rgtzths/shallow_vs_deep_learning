import tensorflow as tf

from datasets.IOT_DNL.iot_dnl import IOT_DNL
from datasets.TON_IOT.ton_iot import TON_IOT
from datasets.UNSW.unsw import UNSW
from datasets.Slicing5G.slicing5g import Slicing5G
from datasets.NetSlice5G.netslice5g import NetSlice5G
from datasets.NetworkSlicing5G.networkslicing5g import NetworkSlicing5G

DATASETS = {
    "NetSlice5G" : NetSlice5G,
    "Slicing5G": Slicing5G,
    "NetworkSlicing5G" : NetworkSlicing5G,
    "IOT_DNL": IOT_DNL,
    "UNSW": UNSW,
    "TON_IOT": TON_IOT,
}