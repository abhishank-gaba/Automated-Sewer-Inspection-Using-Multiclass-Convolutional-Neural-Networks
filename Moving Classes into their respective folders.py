import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from shutil import copyfile
import pickle
def parse_tuple(string):
try:
s = ast.literal_eval(str(string))
if type(s) == tuple:
return s
return
except:
return
# Read data
df = pd.read_csv("../labels.csv", index_col=0, dtype={'filename':str, 'tags': str,
'index': int})
df['tags'] = df['tags'].apply(parse_tuple)
mlb = MultiLabelBinarizer()
mlb.fit(df['tags'].values.tolist())
with open('mlb.pkl', 'wb') as f:
pickle.dump(mlb, f)
if not os.path.isdir('../train_val_images/{}'.format('none')):
os.mkdir('../train_val_images/{}'.format('none'))
for c in mlb.classes_:
if not os.path.isdir('../train_val_images/{}'.format(c)):
os.mkdir('../train_val_images/{}'.format(c))
# Start moving images
for pth, l in zip(df['filename'].values, df['tags'].values):
if sum(mlb.transform([l])[0]) == 0:
copyfile("../raw_images/{}".format(pth),
"../train_val_images/none/{}".format(pth))
elif mlb.transform([l])[0][0] and sum(mlb.transform([l])[0]) == 1:
copyfile("../raw_images/{}".format(pth),
"../train_val_images/{}/{}".format(mlb.classes_[0], pth))
elif mlb.transform([l])[0][1] and sum(mlb.transform([l])[0]) == 1:
copyfile("../raw_images/{}".format(pth),
"../train_val_images/{}/{}".format(mlb.classes_[1], pth))
elif mlb.transform([l])[0][2] and sum(mlb.transform([l])[0]) == 1:Filtering the Normal Scenes Data
Downsizing Dataset
copyfile("../raw_images/{}".format(pth),
"../train_val_images/{}/{}".format(mlb.classes_[2], pth))
elif mlb.transform([l])[0][0] and mlb.transform([l])[0][1]:
copyfile("../raw_images/{}".format(pth),
"../train_val_images/{}/{}".format(mlb.classes_[1], pth))
elif mlb.transform([l])[0][0] and mlb.transform([l])[0][2]:
copyfile("../raw_images/{}".format(pth),
"../train_val_images/{}/{}".format(mlb.classes_[0], pth))
else:
raise ValueError("Error")
