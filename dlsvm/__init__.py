__version__ = '0.0.1'
try:
    import tensorflow as tf
except ModuleNotFoundError:
    msg = 'you must install tensorflow or tensorflow-gpu'
    raise ModuleNotFoundError(msg)
if not str(tf.__version__).startswith('2'):
    msg = 'you must install 2.0 <= tensorflow < 3.0 before'
    raise ValueError(msg)
