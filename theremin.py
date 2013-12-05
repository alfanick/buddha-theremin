#!/usr/bin/env python
import numpy as np
import cv2
import pyo
import sys
from osax import OSAX

MAX_FREQ = 2500.0
MIN_FREQ = 50.0
FREQ_THRESHOLD = 60.0
MUL = 1
MAX_MUL=5
MIN_MUL=0
MUL_THRESHOLD=0.01
MIN_VOLUME=2
MAX_VOLUME=10

class Buddha:

  def __init__(self):
    self.cascade = cv2.CascadeClassifier('classifiers/buddha.xml')
    self.position = None
    self.dimensions = None
    
  def detectAndChoose(self, frame):
    self.buddhas = self.cascade.detectMultiScale(frame, 1.3, 2)
    
    for i,(x,y,w,h) in enumerate(self.buddhas):
      cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
      cv2.putText(frame, "{}".format(i),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

  def choose(self, number):
    if number < len(self.buddhas):
      x,y,w,h = self.buddhas[number]
      self.position = (x,y,)
      self.dimensions = (w,h,)

  def getPosition(self):
    return self.position

  def getCenter(self):
    x,y = self.position
    w,h = self.dimensions
    return (x+w/2, y+h/2,)

  def getDimenstions(self):
    return self.dimensions

  def chosen(self):
    if self.position is not None:
      return True
    else:
      return False

class Player:
  def __init__(self):
    self.server = pyo.Server().boot()
    self.server.start()
    self.signal = pyo.Sig(value=0)
    self.portamento = pyo.Port(self.signal, risetime=.1, falltime=.1)
    self.sine = pyo.Sine(freq=[self.portamento, self.portamento*1.01], mul=MUL).out()

  def freq(self,newFreq=None):
    if newFreq is not None:
      self.signal.setValue(newFreq)
    else:
      return self.signal.value
  def mul(self, newMul=None):
    if newMul is not None:
      self.sine.setMul(newMul)
    else:
      return self.sine.mul

class Tool:
  def __init__(self):
    self.position = None
    self.lower_color = np.array([110,50,50])
    self.upper_color = np.array([130,255,255])
    self.cut = (0,100000,)

  def getPosition(self):
    return self.position

  def setColor(self, lower, upper):
    self.lower_color = lower
    self.upper_color = upper

  def setCut(self, crange):
    self.cut = crange
  
  def detectPosition(self, frame):
    # print self.cut
    res = cv2.multiply(frame, np.array([0.73]))
    res = res[:, self.cut[0]:self.cut[1]]
    res = cv2.blur(res, (3,3))
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
    res = cv2.bitwise_and(res,res, mask= mask)
    res = cv2.erode(res, np.ones((8,8))/2, iterations=3)
    res = cv2.dilate(res, np.ones((8,8))/2, iterations=3)
    res = cv2.Canny(res, 40, 140)
    contours,_ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x = 0; y = 0
    position = None
    for contour in contours:
      moments = cv2.moments(contour)
      if moments['m00']:
        x += int(moments['m10']/moments['m00'])
        y += int(moments['m01']/moments['m00'])
    if len(contours) > 0:
      self.position = (self.cut[0]+x/len(contours), y/len(contours))
    return res

if __name__ == "__main__":
  vc = cv2.VideoCapture(0)
  _, frame = vc.read()

  buddha = Buddha()
  player = Player()
  freq_tool = Tool()
  mul_tool = Tool()
  sa = OSAX()
  
  # Choice of buddha
  cv2.namedWindow("choice")
  while True and not buddha.chosen():
    if frame is not None:
      if not buddha.chosen():
        buddha.detectAndChoose(frame)
        cv2.imshow("choice", frame)
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key >= ord('0') and pressed_key <= ord('9'):
          buddha.choose(pressed_key - ord('0'))
        elif pressed_key == ord('q'):
          sys.exit(0)
    rval, frame = vc.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

  cv2.namedWindow("preview")
  cv2.destroyAllWindows()
  while True:
    if frame is not None:
      # Set frame ranges for tools
      mul_tool.setCut((int(frame.shape[1]/2),frame.shape[1]+100,))
      freq_tool.setCut((0,int(frame.shape[1]/2),))

      # Detect param changing tools
      freq_tool.detectPosition(frame)
      mul_tool.detectPosition(frame)

      # Get positions
      buddha_pos = buddha.getCenter()
      freq_pos = freq_tool.getPosition()
      mul_pos = mul_tool.getPosition()

      if buddha_pos:
        # Change frequency
        if freq_pos:
          newFreq = MIN_FREQ + (MAX_FREQ-MIN_FREQ)*distance(freq_pos, buddha_pos)/1000
          if(abs(player.freq()-newFreq) > FREQ_THRESHOLD): 
            player.freq(int(newFreq))

        # Change volume
        if mul_pos:
          newVolume = MIN_VOLUME + (MAX_VOLUME-MIN_VOLUME)*distance(mul_pos, buddha_pos)/1000
          sa.set_volume(newVolume)

      cv2.imshow("preview", frame)

    rval, frame = vc.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      sys.exit(0)
