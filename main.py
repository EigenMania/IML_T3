#################################
#            IML_T2             #         
#################################
#
# File Name: main.py
#
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz)

import pandas as pd
import tensorflow as tf


train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

data = train['x1']

print(data)
print(type(data))


