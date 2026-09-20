
import numpy as np
from dotenv import load_dotenv
import os
import pickle
from scapy.all import rdpcap
from scapy.plist import PacketList
from scapy.error import Scapy_Exception

##################### GLOBALS #################################################

load_dotenv()
base_dir = save_pick = read_dir = None


## Load data dirs from BASE_DIR_<NAME> / SAVE_PROC_DIR_<NAME> / READ_PROC_DIR_<NAME> in .env
def use_profile(name):
    global base_dir, save_pick, read_dir
    # base_dir = os.path.expanduser(os.environ[f"BASE_DIR_{name}"])
    # save_pick = os.path.expanduser(os.environ[f"SAVE_PROC_DIR_{name}"])
    read_dir = os.path.expanduser(os.environ[f"READ_PROC_DIR_{name}"])


## Prefix and Suffix of files
base_file_name_pre = "acks_mcs_"
base_file_name_suff = ".pcap"

helper_file_name_pre = "ack_MCS"
helper_file_name_suff = "_HELPER.pcap"

def has_valid_fcs_24(pkt):
    raw = bytes(pkt)
    last_byte = raw[-1]
    fcs_ok = (last_byte & 0x80)
    return fcs_ok != 0

def calc_correct_pkts(frames: PacketList):
    num_valid_acks = 0
    for pkt in frames:
        if(has_valid_fcs_24(pkt) == False):
           continue
        num_valid_acks += 1
    return num_valid_acks

def calc_helper_ack_num(frames: PacketList):
    return len(frames) ## It is assumed that the helper WiFi card throws away frames with baf fcs

def get_acks_mcs(device_dir_name:str, capture_dir_name:str, num_mcs=8):
    full_path = base_dir + '/'  + device_dir_name + '/' + capture_dir_name + '/'
    num_acks = []
    for i in np.arange(num_mcs):
        curr_file_path = full_path + '/' + base_file_name_pre + str(i) + base_file_name_suff
        try:
            pcap_file = rdpcap(curr_file_path)
            num_acks.append(calc_correct_pkts(pcap_file))
        except (FileNotFoundError, Scapy_Exception):
            # print(f"\n Warning!: No File:{curr_file_path}\n")
            num_acks.append(-1) ## Note -1 means we do not have that data
    return num_acks

def get_acks_help(device_dir_name:str, capture_dir_name:str, num_mcs=8):
    full_path = base_dir + '/'  + device_dir_name + '/' + capture_dir_name + '/'
    num_acks_helper = []
    for i in np.arange(num_mcs):
        curr_file_path = full_path + '/' + helper_file_name_pre + str(i) + helper_file_name_suff
        try:
            pcap_file = rdpcap(curr_file_path)
            num_acks_helper.append(calc_helper_ack_num(pcap_file))
        except FileNotFoundError:
            # print(f"No Helper:{curr_file_path}")
            num_acks_helper.append(-1)
    return num_acks_helper
     

def get_acks_dec(device_dir_name:str, num_mcs=8):
    all_dec = []
    all_help = []
    for j in range(5,161,5):
        dir_cap_name = 'minus_' + str(j) + '_stf/'
        all_dec.append(get_acks_mcs(device_dir_name, dir_cap_name, num_mcs))
        all_help.append(get_acks_help(device_dir_name, dir_cap_name, num_mcs))
    return all_dec,all_help

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
class MY_CAP:
    def __init__(self, device_dir_name:str, num_mcs, mft="None"):
        self.name = device_dir_name
        self.base_acks = []
        self.base_acks_helper=[]
        self.dec_acks = []
        self.dec_acks_helper=[]
        self.num_mcs = num_mcs
        self.manufacturer = mft

    def process_acks(self):
        self.base_acks = get_acks_mcs(self.name, "stock", self.num_mcs)
        self.base_acks_helper = get_acks_help(self.name, "stock", self.num_mcs)
        self.dec_acks, self.dec_acks_helper = get_acks_dec(self.name, self.num_mcs)
       

    def save_obj(self, file_name:str):
        f_name_full = save_pick + '/' + file_name
        with open(f_name_full, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load(cls, file_name: str):
        f_name_full = read_dir + '/' + file_name
        with open(f_name_full, "rb") as f:
            return pickle.load(f)