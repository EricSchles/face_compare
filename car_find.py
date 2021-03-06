import cv2
import matplotlib.pyplot as plt

cctv_image = cv2.imread('cars.jpg')

# the 'cascade.xml' file is the file generated by the training script above
vehicle_classifier = cv2.CascadeClassifier('cascade.xml')

# various parameters can be passed to modify how objects are detected
vehicles = vehicle_classifier.detectMultiScale(cctv_image, 1.1, 2, maxSize=(200,200))

print 'Vehicles detected: %d' % (len(vehicles))

# draw a rectangle around every vehicle detected
for (x,y,w,h) in vehicles:
    cv2.rectangle(cctv_image, (x,y), (x+w, y+h),(255,0,0),2)

plt.figure(figsize=(9,9))
plt.axis('off')
plt.imshow(cv2.cvtColor(cctv_image, cv2.COLOR_BGR2RGB))
