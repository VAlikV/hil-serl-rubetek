from collections import OrderedDict
import pickle


with open('demo_data/rozum_push_3_demos_2025-11-15_20-23-36.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[0])