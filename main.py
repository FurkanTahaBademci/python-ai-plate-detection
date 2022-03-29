from PIL import Image
import pytesseract
import cv2
import os
from datetime import datetime 
import time 
from PyQt5.QtWidgets import*
from framework import Ui_MainWindow
from PyQt5.QtGui import*
from PyQt5.QtCore import*
from PyQt5.QtWidgets import*
import numpy as np



class untitled_python(QMainWindow):
    
    def __init__(self):
    
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #---FPS---#
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        #-- DEĞİŞKENLER --#
        self.numara = 1000  
        self.a = 0
        self.text_sabit = ""
        self.text_degisken = ""

#--------------------------- GÖSTERME FOKSİYONLARI ---------------------------------#
    def resim_gosterme_label(self,image):

        en_boy = self.ui.label.geometry()
        w,h = en_boy.getRect()[2:]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_orjinal = cv2.resize(image,(w+5,h))
        ConvertToFormat = QImage(image_orjinal.data, image_orjinal.shape[1], image_orjinal.shape[0], QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(ConvertToFormat))

    def plaka_gosterme_label(self,image):

        en_boy = self.ui.label_2.geometry()
        w,h = en_boy.getRect()[2:]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_orjinal = cv2.resize(image,(w+5,h))
        ConvertToFormat = QImage(image_orjinal.data, image_orjinal.shape[1], image_orjinal.shape[0], QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QPixmap.fromImage(ConvertToFormat))

    def plaka_gosterme_theresh(self, thereshold):

        en_boy = self.ui.label_2.geometry()
        w,h = en_boy.getRect()[2:]
        image_orjinal = cv2.resize(thereshold,(w+5,h))
        ConvertToFormat = QImage(image_orjinal.data, image_orjinal.shape[1], image_orjinal.shape[0], QImage.Format_Grayscale8)
        self.ui.label_3.setPixmap(QPixmap.fromImage(ConvertToFormat))

    #------------------- KAMERA AÇ KAPA FOKSİYONLARI ---------------------#

    def kamera_ac(self):

        self.a = 1
        self.ui.push2.setEnabled(True)
        self.ui.push.setEnabled(False)
        self.kamera_ac_video()

    def kamera_kapat(self):

        self.a = 0
        self.ui.push.setEnabled(True)
        self.ui.push2.setEnabled(False)


    def kamera_ac_video(self):

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #---------PLAKA ALGILAMA -----------------#
        with open('coco.names', 'r') as f:
            classes = f.read().splitlines()
        net = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg', 'custom.weights')
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

        cap = cv2.VideoCapture("video_trafik.mp4")

        while self.a == 1:

            ret, img = cap.read()
            # img = cv2.flip(img, 1)
            img_copy = img.copy()

            if ret:
                classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

                #------------- ÇİZİM VE ROİ ALANI  ------------------#
                for (classId, score, box) in zip(classIds, scores, boxes):

                    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)

                    x,y,w,h = box[0], box[1], box[2], box[3]
                    plaka = img_copy[y:y+h,x:x+w]       # PLAKA ROİ ALANI
                    plaka = cv2.resize(plaka, (300, 80))
                    text = '%s: %s%.2f' % (classes[classId],"%", 100 * score)
                    cv2.putText(img, text, (box[0], box[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)

                    #-------------- PLAKA ------------------#
                    hsv = cv2.cvtColor(plaka,cv2.COLOR_BGR2HSV)
                    v_degeri = cv2.split(hsv)[2]

                    #--KESKİNLEŞTİRME---#
                    kernel = np.array(
                    [[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])
                    sharpened = cv2.filter2D(v_degeri, -1, kernel)
                    median_blur = cv2.medianBlur(sharpened,5)   # MEDIAN BLUR
                    thereshold = cv2.threshold(median_blur, 60, 240, cv2.THRESH_OTSU)[1]
                    filename = "{}.png".format(os.getpid())  # dosyanın ismini çekiyoruz

                    # cv2.imwrite(filename, thereshold)        # dosyayı kaydediyoruz
                    cv2.imwrite(filename, v_degeri)        # dosyayı kaydediyoruz
                                                           #********** dosya yolu düzenlenmeli ********
                    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # görmediği için manuel açıyoruz
                    self.text_sabit = pytesseract.image_to_string(Image.open(filename))
                    os.remove(filename)   
                    

                    
                    if 10>len(self.text_sabit)>5 :
                        print("[  PLAKA  ]: ", self.text_sabit)
                        self.text_degisken = self.text_sabit
                        self.ui.label_4.setText(self.text_sabit) 
                        
                try:
                    self.resim_gosterme_label(img) # ORJINAL GÖRÜNTÜ 
                    self.plaka_gosterme_label(v_degeri)   # PLAKA ROİ
                    self.plaka_gosterme_theresh(thereshold) # PLAKA ROİ THRESHOLD

                except:
                    pass
                            
                if self.a == 0: 
                    self.ui.label.clear() 
                    self.ui.label_2.clear()
                    self.ui.label_3.clear()

            #------- FPS HESAPLAMA ------#
            self.new_frame_time = time.time()
            self.fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            self.fps = round(self.fps,3)
            self.ui.fps.setText(str(self.fps))
            
            cv2.waitKey(1)
         
def arayuz_ac():

    uygulama = QApplication([])
    pencere = untitled_python()
    pencere.show()
    uygulama.exec_()

arayuz_ac()

