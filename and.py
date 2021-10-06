from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model
import pandas as pd
import numpy  as np
import logging as lg;
import os;
log_dir ='logs'
os.makedirs(log_dir,exist_ok=True)
logging_str = "[%(asctime)s:%(levelname)s:%(name)s]%(message)s"
lg.basicConfig(filename=os.path.join(log_dir,'and.log'),level  = lg.INFO, format = logging_str)


AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)


lg.info("thedata frame is {df}")

x,y = prepare_data(df)

ETA = 0.3 # 0 and 1
epoch = 10

model = Perceptron(eta=ETA, epoch=epoch)
model.fit(x, y)

_ = model.total_loss()

save_model(model,filename='and.model')