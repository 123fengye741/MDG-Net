#from baseline import get_unet
#from baseline import newmyunet
from SEatention import newmyunet
#from backbonenon import baseunet
from glob import glob
from PIL import Image
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import label
#from pycocotools import mask as maskUtils
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
import scipy.misc
import imageio
import cv2
from keras.layers import ReLU
from sklearn.metrics import roc_auc_score,roc_curve
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batchsize = 1
input_shape = (576, 576)


def batch(iterable, n=batchsize):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def read_input(path):
    x = np.array(cv2.imread(path))/255.
    return x


def read_gt(path):
    x = np.array(Image.open(path))
    return x[..., np.newaxis]/np.max(x)

if __name__ == '__main__':
    model_name = "baseunet"

    weight_decay=1e-4
    val_data = list(zip(sorted(glob('E:\\maching learning\\data\\DRIVE\\test\\images\\*.tif')),
                          sorted(glob('E:\\maching learning\\data\\DRIVE\\test\\2nd_manual\\*.gif')),
                         sorted(glob('E:\\maching learning\\data\\DRIVE\\test\\mask\\*.gif'))))

   # try:
      #  os.makedirs("/home/njj/deep/vessel_image_segmentation-master/out/"+model_name+"test/", exist_ok=True)
   # except:
      #  pass
    print(val_data)
    #model = get_unet(do=0.1, activation=ReLU)
    model = newmyunet(576,576,3)
    #model = baseunet (576,576,3)
   # model.summary()

    file_path = model_name + ".weights.h5"

    model.load_weights(file_path, skip_mismatch=False)

    gt_list = []
    pred_list = []
    
    print(len(val_data))

    for batch_files in tqdm(batch(val_data), total=len(val_data)//batchsize):
      #  print(batch_files)

        imgs = [resize(read_input(image_path[0]), input_shape) for image_path in batch_files]
        seg = [read_gt(image_path[1]) for image_path in batch_files]
        mask = [read_gt(image_path[2]) for image_path in batch_files]

        imgs = np.array(imgs)

        pred = model.predict(imgs)
        a=pred[0]
       # a=a.reshape(576,576,1)
      #  plt.imshow(a)
        #scipy.misc.imsave("1.jpg",a)
         
        #cv2.imwrite("1.jpg", a)
        
        pred_all = (pred)

        pred = np.clip(pred, 0, 1)

        for i, image_path in enumerate(batch_files):

            pred_ = pred[i, :, :, 0]

            pred_ = resize(pred_, (584, 565))

            mask_ = mask[i]

            gt_ = (seg[i]>0.5).astype(int)

            gt_flat = []
            pred_flat = []

            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if mask_[p,q]>0.5: # Inside the mask pixels only
                        gt_flat.append(gt_[p,q])
                        pred_flat.append(pred_[p,q])

            print(pred_.size, len(gt_list))

            gt_list += gt_flat
            pred_list += pred_flat

            pred_ = 255.*(pred_ - np.min(pred_))/(np.max(pred_)-np.min(pred_))

            image_base = image_path[0].split("\\")[-1]

            imageio.imsave("E:\\maching learning\\DRIVE 98.31\\test1\\"+image_base, pred_)
            plt.imshow(pred_)
       # print(len(pred))
  #  print(len(gt_list), len(pred_list))
    threshold_confusion=0.5
    y_pred = np.empty(len(pred_list))
    for i in range(len(pred_list)):
        if pred_list[i]>=threshold_confusion:
             y_pred[i]=1
        else:
             y_pred[i]=0
    #confusion = confusion_matrix(y_true, y_pred)
    confusion = confusion_matrix(gt_list, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
         accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print("Global Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print("Sensitivity: " +str(sensitivity))
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
       precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print("Precision: " +str(precision))
    
    path_experiment = "E:\\maching learning\\DRIVE 98.31\\"
#AUC ROC = 0
    AUC_ROC = roc_auc_score(gt_list, pred_list)
    
    print("AUC_ROC: " +str(AUC_ROC))
    fpr, tpr, thresholds = roc_curve (gt_list, pred_list)
    file_perf = open(path_experiment+'performances.txt', 'w')
    file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
    file_perf.close()
    
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    roc_curve =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment+"ROC.png")