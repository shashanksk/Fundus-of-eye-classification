from keras.preprocessing.image import save_img
import os,time,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input
from data import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import keras.backend as K
from keras.models import Model
from pathlib import Path

def recall_m(y_true, y_pred):
    y_pred = K.reshape(y_pred, shape= ((K.shape(y_pred)[0]*K.shape(y_pred)[1]*K.shape(y_pred)[2]),num_classes))
    y_true = K.reshape(y_true, shape= ((K.shape(y_true)[0]*K.shape(y_true)[1]*K.shape(y_true)[2]),num_classes))
    true_positives = K.sum(K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)[0:-1]) # Without background, which is the final class
    possible_positives = K.sum(K.sum(K.round(K.clip(y_true, 0, 1)),axis=0)[0:-1])
    recall = true_positives / (possible_positives+ K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    y_pred = K.reshape(y_pred, shape= ((K.shape(y_pred)[0]*K.shape(y_pred)[1]*K.shape(y_pred)[2]),num_classes))
    y_true = K.reshape(y_true, shape= ((K.shape(y_true)[0]*K.shape(y_true)[1]*K.shape(y_true)[2]),num_classes))
    true_positives = K.sum(K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)[0:-1])
    predicted_positives = K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)),axis=0)[0:-1])
    precision = true_positives / (predicted_positives+ K.epsilon())
    return precision
def fscore(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+ K.epsilon()))
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
def colour_code(image, label_values):
    x = np.argmax(image, axis = -1)
    colour_codes = np.array(label_values)
    x = colour_codes[x.astype(int)]
    return x

magnification1 = "4x"
magnification2 = "semi"
magnification3 = "40x"
magnification4 = "10x"
if sys.argv[1:]:
    magnification = sys.argv[1]
path = os.path.dirname(os.getcwd())
path = Path(path)
path2 = Path("/mnt/smiledata/shajahan_trails/Weights")
predict_folder = path/"Data_keras"/magnification1/"1"
#weights_folder = path/"Weights"/magnification4/"weights"#_resunet"
weights_folder = path2/magnification1/"weights"
output_folder = path/"Results/Predictions"/magnification1/"journ"
model_name = path/"Modelh5/core_model.h5"
#model_name = path/"Modelh5/resunet.h5"

#model = load_model(checkpoint_path, custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef,'iou': iou})
model = load_model(str(model_name), custom_objects={'fscore': fscore})
#model.load_weights("%s/weights.194.hdf5"%weights_folder) #10X
model.load_weights("%s/weights.186.hdf5"%weights_folder)  #4X
start = time.time()

#newInput = Input(shape=(1440,1920,3))    # let us say this new InputLayer
#newInput = Input(shape=(3584,4800,3))    # let us say this new InputLayer
newInput = Input(shape=(256,256,3))    # let us say this new InputLayer
#newInput = Input(shape=(1600,1600,3))    # let us say this new InputLayer
newOutputs = model(newInput)
model1 = Model(newInput, newOutputs)
label=True
if label:
    val_x,val_y,img_name = validation3(str(predict_folder/"images"),str(predict_folder/"mask"))
    val_y = colour_code(val_y,label_values)
    img_type = "gt"
else:    
    val_x,img_name = validation_jpg(str(predict_folder/"images"))
    val_y = val_x
    img_type = "org"

num_val = len(img_name)
z= model1.predict(val_x, batch_size=1, verbose=0, steps=None)

if os.path.isdir(output_folder) is not True:
    os.mkdir(output_folder)


z = colour_code(z,label_values)
for i in range(len(z)):
    a = img_name[i]
    image_name = os.path.basename(a)
    print(image_name)
    out = z[i]
    inp = val_y[i]
    image_name = image_name[:image_name.index('.')] 
    save_img('%s/%s_pred.png'%(output_folder,image_name), out)
    save_img('%s/%s_%s.png'%(output_folder,image_name,img_type), inp)
end = time.time()
print(f"time taken for { len(z) } images is {end - start} seconds ")
