import sys
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppressing cpu messages

# **** set font colors where needed ****
yellow =  '\033[93m'
endcolor = '\033[0m'
red = '\033[91m'

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

# **** computation block ****
with tf.device(device_name): # tf. dvice sets device as CPU or GPU
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

# **** tf.session block *****
startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session: #False suppresses GPU messages or displays them True
        result = session.run(sum_operation)
        print("\n" *2)
        print(yellow + "Actual result from computation --> " + endcolor,result)

# Print output with spaces fro clarity
print("\n")
print("Tensor Shape:", shape, "Device used:", red + device_name + endcolor)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)