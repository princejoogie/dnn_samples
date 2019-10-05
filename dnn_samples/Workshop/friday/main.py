import cv2
import glob as gb
import numpy as np

images = gb.glob('leaves/*/*')
def notes():
    ''' Resize '''
    #img = cv2.resize(img, None, fx=0.5, fy=0.5)
    ''' White Balance '''
    # iWhite = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    ''' Gaussian Blue '''
    # Ibin = cv2.GaussianBlur(I, (5,5), 0)
    ''' Equalized Histogram '''
    # iWhite[:,:,0] = cv2.equalizeHist(iWhite[:,:,0])
    ''' Binarization '''
    # Ibin = cv2.adaptiveThreshold(Ir, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

def PreProcess(image_name):
    ''' Binarize '''
    I = cv2.imread(image_name)
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Ir = cv2.resize(Igray, None, fx=0.5, fy=0.5)
    ret, Ibin = cv2.threshold(Igray, 100, 255, cv2.THRESH_BINARY)
    return Ibin

def main():
    for fileName in images:
        Iprep = PreProcess(fileName)
        newFile = 'training' + fileName[6:]
        print(newFile)
        cv2.imwrite(newFile, Iprep)

if __name__ == '__main__':
    main()
    print(cv2.__version__)