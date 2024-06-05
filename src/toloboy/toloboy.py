import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
# from torch.utils.data import Dataset #NOTE: installing pytorch with poetry is a pain in the ass!

####################
#### functions #####
####################

############################################
#### Color space transformers functions ####
############################################

def RGB2LAB2(R0, G0, B0):
    """
    convert RGB to the personal LAB (LAB2) 
    the input R,G,B,  must be 1D from 0 to 255 
    the outputs are 1D  L [0 1], a [-1 1] b [-1 1]
    """

    R=R0/255
    G=G0/255
    B=B0/255
    
    
    Y=0.299*R + 0.587*G + 0.114*B
    X=0.449*R + 0.353*G + 0.198*B
    Z=0.012*R + 0.089*G + 0.899*B  
        
    L = Y
    a = (X - Y)/0.234
    b = (Y - Z)/0.785
    
    return L, a, b



def LAB22RGB(L, a, b):
    """
    convert the personal LAB (LAB2)to the RGB 
    the input L,a,b,  must be 1D L [0 1], a [-1 1] b [-1 1]
    the outputs are 1D  R g B [0 255]
    """
    
    a11 = 0.299
    a12 = 0.587
    a13 = 0.114
    a21 = (0.15/0.234)
    a22 = (-0.234/0.234)
    a23 = (0.084/0.234)
    a31 = (0.287/0.785)
    a32 = (0.498/0.785)
    a33 = (-0.785/0.785)
    
    aa=np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    C0=np.zeros((L.shape[0],3))
    C0[:,0]=L[:,0]
    C0[:,1]=a[:,0]
    C0[:,2]=b[:,0]
    C = np.transpose(C0)
    
    X = np.linalg.inv(aa).dot(C)
    X1D=np.reshape(X,(X.shape[0]*X.shape[1],1))
    p0=np.where(X1D<0)
    X1D[p0[0]]=0
    p1=np.where(X1D>1)
    X1D[p1[0]]=1
    Xr=np.reshape(X1D,(X.shape[0],X.shape[1]))
    
    Rr = Xr[0][:]
    Gr = Xr[1][:]
    Br = Xr[2][:]
    
    R = np.uint(np.round(Rr*255))
    G = np.uint(np.round(Gr*255))
    B = np.uint(np.round(Br*255))
    
    return R, G, B



def from_LAB_to_RGB_img(L, AB):
    """
    Takes the L and AB channels retunred from the transformation and 
    convert the image to RGB colorspace.
    """

    # print("*** Transforming LAB img to RGB ***")    
    x_dim, y_dim = L.shape[0], L.shape[1]
    predicted_RGB=np.uint8(np.zeros((x_dim,y_dim,3)))
    AB = np.squeeze(AB)
    # print(f"Shape o AB in conversion is {AB.shape}")
    a0, b0 = AB[:, :, 0], AB[:, :, 1]

    # print(f"{np.squeeze(L).shape}, {a0.shape}, {b0.shape}")

    Rr, Gr, Br = LAB22RGB(L.reshape(-1, 1), 
                          a0.reshape(-1, 1), 
                          b0.reshape(-1, 1))
 
    predicted_RGB[:, :,0] = np.reshape(Rr,(x_dim,y_dim))
    predicted_RGB[:, :,1] = np.reshape(Gr,(x_dim,y_dim))
    predicted_RGB[:, :,2] = np.reshape(Br,(x_dim,y_dim))


    return predicted_RGB



def plot_multiple_imgs(orig_img, imgs_ls, with_orig=True, col_title=None, img_size = 10, font_s = 12, **imshow_kwargs):
    """
    Usefull to compare a reference image with list of additional images.
    """

    if not isinstance(imgs_ls[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs_ls = [imgs_ls]

    num_rows = len(imgs_ls)
    num_cols = len(imgs_ls[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize = (img_size, img_size))
    for row_idx, row in enumerate(imgs_ls):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # if with_orig:
    #     axs[0, 0].set(title='Original image')
    #     axs[0, 0].title.set_size(8)
    
    
    if col_title is not None:
        # for row_idx in range(num_rows):
        #     axs[row_idx, 0].set(ylabel=col_title[row_idx])
        for ax, title in zip(axs.flatten(), col_title):
            ax.set_title(f"{title}", fontsize=font_s)
    else:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(font_s)

    plt.tight_layout()



############################
#### Metrics functions ####
###########################


def psnr(img1, img2):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def mse(imageA, imageB, nband):
	"""
    the 'Mean Squared Error' between the two images is the
	sum of the squared difference between the two images;
	NOTE: the two images must have the same dimension
    """
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



# def mae(imageA, imageB, bands):
# 	"""
#     the 'Mean Absolute Error' between the two images is the
# 	sum of the squared difference between the two images;
# 	NOTE: the two images must have the same dimension
#     """
# 	err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
# 	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    
#     return err




def rmse(imageA, imageB, nband):
	# the 'Root Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
	err = np.sqrt(err)
	return err


##############################
#### Custom data classes #####
##############################

# the idea was borrowed from here: https://towardsdatascience.com/custom-dataset-in-pytorch-part-1-images-2df3152895

class Swisstopodataset(Dataset):
    def __init__(self, img_indx, transform = None, large_dataset=False, return_label = True):
        self.img_indx = img_indx
        self.transform = transform
        self.large_dataset = large_dataset
        self.return_label = return_label
        # self.augment = augment

    def __len__(self):
        return len(self.img_indx)

    def __getitem__(self, idx):
        # img_filepath  = self.img_indx[idx]
        if self.large_dataset:
            port = 1986
        else:
            port = 1985
        raw_data_csv_file_link = f"https://perritos.myasustor.com:{port}/metadata.csv"
        metadata_file = pd.read_csv(raw_data_csv_file_link, index_col=0)
        img_in_server_link = f"https://perritos.myasustor.com:{port}/data/img_id_{self.img_indx[idx]}.jpg"
        response = requests.get(img_in_server_link)
        image = Image.open(BytesIO(response.content))
        # image = cv2.imread(img_filepath)
        # image = Image.fromarray(image)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # define here a label if existing
        label = metadata_file["class"].iloc[idx]

        if self.transform is not None:
            # print ("*** Transforming RGB img to LAB ***")
            image = self.transform(image)
        
        # if self.augment is not None:
        #     # print ("*** Applying augmentation on L channel ***")
        #     image_L, imageAB = image
        #     image_L = self.augment(image_L)
        #     image = (image_L, imageAB)


        if self.return_label:
            return image, label
        else:
            return image
