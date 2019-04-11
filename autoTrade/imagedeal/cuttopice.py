import cv2
import numpy as np
import pandas as pd

test_path=r"C:\Users\Administrator\Desktop\autoTradeDataSet\1.png"

mat_image=cv2.imread(test_path)
mat_image_cut_noinfo=mat_image[:,:1849]
mat_image_cut_noinfo_resize1776=cv2.resize(mat_image_cut_noinfo,(1776,1058))
cv2.imshow('cut',mat_image_cut_noinfo_resize1776)

# image_cut_gray=cv2.cvtColor(mat_image_cut_noinfo_resize1776,cv2.COLOR_BGR2GRAY)
# cv2.imshow('cut_gray',image_cut_gray)

# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
# image_cut_gray_erode=cv2.erode(image_cut_gray,kernel)
# cv2.imshow('cut_gray_erode',image_cut_gray_erode)
#
# kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
# image_cut_gray_erode_dilate=cv2.dilate(image_cut_gray_erode,kernel1)
# cv2.imshow('image_cut_gray_erode_dilate',image_cut_gray_erode_dilate)
cv2.waitKey()

class ImageDeal:
    def read_resize(self,path):
        mat_image = cv2.imread(path)
        mat_image_cut_noinfo = mat_image[:, :1849]
        mat_image_cut_noinfo_resize1776 = cv2.resize(mat_image_cut_noinfo, (1776, 1058))
        return mat_image_cut_noinfo_resize1776

    def get_all_label(self,path):
        data=pd.read_csv(path)
        return data.values

    def get_image_label(self,path_image,path_csv):
        image=self.read_resize(path_image)
        labels=self.get_all_label(path_csv)

        #28宽度，4个未来label
        for i in range(196-28-4):
            pass

