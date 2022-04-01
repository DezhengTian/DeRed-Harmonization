import tensorflow as tf
import DeRed
import os
if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    for _ in ['transverse','coronal','sagittal']:
        test_DeRed = DeRed.DeRed(_)
        test_DeRed.predict()




