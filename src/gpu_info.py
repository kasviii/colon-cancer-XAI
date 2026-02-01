# gpu_info.py
import tensorflow as tf, json, os, sys

print("Python:", sys.version.splitlines()[0])
print("TensorFlow version:", tf.__version__)

try:
    info = tf.sysconfig.get_build_info()
    print("\n tf.sysconfig.get_build_info():")
    print(json.dumps(info, indent=2))
except Exception as e:
    print("\nCould not get build info:", e)

print("\ntf.test.is_built_with_cuda():", tf.test.is_built_with_cuda())
print("tf.config.list_physical_devices('GPU'):", tf.config.list_physical_devices('GPU'))

print("\nEnvironment variables (CUDA_PATH and PATH entries containing 'CUDA'):")
print("CUDA_PATH =", os.environ.get("CUDA_PATH"))
# Print any PATH pieces containing CUDA
path = os.environ.get("PATH","")
matches = [p for p in path.split(";") if "CUDA" in p.upper() or "CUDNN" in p.upper()]
print("PATH entries with CUDA/cudnn:", matches)
