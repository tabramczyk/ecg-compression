import tensorflow as tf
import wfdb
import matplotlib.pyplot as plt
import numpy as np

patientID = str(112)
"""
errs = []
for i in range(100, 235):
  try:
    record = wfdb.rdsamp(str(i)) #; print(record)
  except:
    errs.append(i)
print(len(errs))
print(errs)
"""
record = wfdb.rdsamp(patientID); print(record)

data = np.array(record[0]); np.shape(data)

from scipy.signal import argrelextrema
import more_itertools as mit

def bigmaximum(x):
  return argrelextrema(x, np.greater_equal, order=120)


# get "really big" maxima
maxima = (bigmaximum(data[:,0])[0], bigmaximum(data[:,1],)[0])

# max at the very end is not true max
maxima = tuple(map(lambda x: x[:-1] if x[-1] == np.size(data,0)-1 else x,
                   maxima))

# get real spans (exclude flats --- they should be no longer than 15)
spans = map(np.diff, maxima)
spans = tuple(map(lambda x: x[x > 15], spans))

# reconstruct maxima (some were removed)
maxima = tuple(map(lambda x,y: np.concatenate(([x[0]], y)), maxima, spans))
maxima = tuple(map(np.cumsum, maxima))
Nmaxima = min(np.size(maxima[0]), np.size(maxima[1]))

# calc stats
min_span = tuple(map(np.min, spans))
max_span = tuple(map(np.max, spans))
mean_span = tuple(map(np.mean, spans))
median_span = tuple(map(np.median, spans))
p98_span = tuple(map(lambda x: np.percentile(x, 98), spans))

# show stats
print(f"Maximums:  {Nmaxima}")
print("Span:")
print(f"  min:     ({min_span[0]:3.0f}, {min_span[1]:3.0f})")
print(f"  max:     ({max_span[0]:3.0f}, {max_span[1]:3.0f})")
print(f"  mean:    ({mean_span[0]:3.0f}, {mean_span[1]:3.0f})")
print(f"  median:  ({median_span[0]:3.0f}, {median_span[1]:3.0f})")
print(f"  p98:     ({p98_span[0]:3.0f}, {p98_span[1]:3.0f})")

# rp = plt.plot(data[0:500,:], linewidth=1); plt.show(rp)

from scipy.ndimage import gaussian_filter
"""data_fil = gaussian_filter(data[0:500,0], sigma=2)
data_fil2 = gaussian_filter(data[0:500,1], sigma=2)
plt.plot(data_fil, linewidth=1)
plt.plot(data_fil2, linewidth=1)
plt.show()
"""


def betweenMaxima(data, idx):
  # dla pierwszego i ostatniego to zaslepka
  max00 = maxima[0][idx-1] if idx != 0 else -maxima[0][idx]
  max10 = maxima[1][idx-1] if idx != 0 else -maxima[1][idx]
  max01 = maxima[0][idx]
  max11 = maxima[1][idx]
  max02 = maxima[0][idx+1] if idx+1 != len(maxima[0]) else 2*np.size(data,0) - maxima[0][-1]
  max12 = maxima[1][idx+1] if idx+1 != len(maxima[1]) else 2*np.size(data,0) - maxima[1][-1]

  x01 = (max00 + max01) // 2
  x02 = (max01 + max02) // 2
  x11 = (max10 + max11) // 2
  x12 = (max11 + max12) // 2

  x1 = x01 #(x01 + x11) // 2
  x2 = x02 #(x12 + x12) // 2
  
  sli0 = data[x1:x2, 0]
  sli1 = data[x1:x2, 1]
  return (sli0, sli1)
"""
# note: 0-th and (size-1)-th is special as it doesn't have to be centered
for i in range(0, 5):
  (s0, s1) = betweenMaxima(data, i)
  rp = plt.plot(s0, linewidth=1)
  rp = plt.plot(s1, linewidth=1)
  plt.show(rp)
"""
from math import floor


def rebin(a, size):
  asize = np.size(a)
  a = gaussian_filter(a, sigma=0.2)
  if size == asize:
    return a
  else:
    aa = np.zeros(size)
    aa[0] = a[0]; aa[-1] = a[-1]
    step = asize / size
    for i in range(1, size):
      j = i * step
      ji = int(floor(j))
      jd = j - ji
      if ji+1 == asize:
        aa[i] = a[-1]
      else:
        aa[i] = (1-jd) * a[ji] + jd * a[ji+1]
        #if aa[i] / a[ji] < 0.8 or aa[i] / a[ji+1] < 0.8:
        #  print(i, j, aa[i], a[ji], a[ji+1])
    return aa

onesize = 256 #int((mean_span[0] + mean_span[1]) / 2)
print(f"onesize: {onesize}")

def rebiToMean(a, size=None):
  return tuple(map(lambda x: rebin(x, size or onesize), a))

# some examples

def plot_pair(p0, p1):
  rp = plt.plot(p0, linewidth=1)
  rp = plt.plot(p1, linewidth=1)
  plt.show(rp)
"""
for i in range(0,5):
  (s0, s1) = betweenMaxima(data, i)
  siz = (np.size(s0) + np.size(s1)) // 2
  (r0, r1) = rebiToMean((s0,s1))
  (b0, b1) = rebiToMean((r0,r1), siz)
  plot_pair(s0,s1)
  plot_pair(r0,r1)
  plot_pair(b0,b1)
  
  # test
  siz = (np.size(s0) + np.size(s1)) // 2
  (b0, b1) = rebiToMean((r0,r1), siz)
  siz_back = (np.size(b0) + np.size(b1)) // 2
  print("convert back?")
  print(f"  {siz} -> {onesize} -> {siz_back}")
  ss = (s0,s1)
  ss_back = (b0, b1)
  mse = tuple(map(lambda x,y: ((x - y) ** 2).mean(), ss, ss_back))
  print(f"  mse = {mse}")
"""

# mean mse for ALL

mean_mse = (0,0)
for i in range(1, Nmaxima):
  (s0, s1) = betweenMaxima(data, i)
  ss = (s0,s1)
  siz = (np.size(s0) + np.size(s1)) // 2
  (r0, r1) = rebiToMean(ss)
  (b0, b1) = rebiToMean(ss, siz)
  ss_back = (b0, b1)
  siz_back = (np.size(b0[0]) + np.size(b0[1])) // 2
  mse = tuple(map(lambda x,y: ((x - y) ** 2).mean(), ss, ss_back))
  mean_mse = tuple(map(lambda x,y: x+y, mean_mse, mse))
mean_mse = tuple(map(lambda x: x / (Nmaxima-2), mean_mse))
print(f"mean MSE: {mse}")
#(7.519217867372877e-05, 2.190551662929528e-05)
#(1.1474338535399689e-05, 3.3248595682143493e-06)


###############################################################################
################################ Party Zone ###################################
###############################################################################

import tensorflow.contrib.layers as lays
# from tensorflow.tools.api.generator.api.nn import conv1d
from tensorflow.contrib.nn import conv1d_transpose

epoch_num = 5
lr = 0.001

def autoencoder(inputs):
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], 16, 16, 1])
    # 16 x 16 x 1   ->  8 x 8 x 16
    # 8 x 8 x 16    ->  4 x 4 x 8
    # 4 x 4 x 8     ->  2 x 2 x 4
    net = lays.conv2d(inputs, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 4, [5, 5], stride=2, padding='SAME')
    # decoder
    # 2 x 2 x 4     ->  4 x 4 x 8
    # 4 x 4 x 8     ->  8 x 8 x 16
    # 8 x 8 x 16    ->  16 x 16 x 1
    net = lays.conv2d_transpose(net, 8, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    net = tf.reshape(net, [tf.shape(inputs)[0], 256, 1]) 
    return net

ae_inputs = tf.placeholder(tf.float32, (None, 256, 1))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()
"""
with tf.Session() as sess:
    sess.run(init)
   """ 
