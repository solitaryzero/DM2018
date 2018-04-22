import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model
import genFrame as gf

res = gf.genFrames(1,20)
res_aq = res[0]
res_me = res[1]
print(res_aq[0])
print(res_me[0])