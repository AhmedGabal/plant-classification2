import flask
import werkzeug
import time
from flask import request, jsonify
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
from keras.models import load_model
from keras import backend as K
app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def classify():
	files_ids = list(flask.request.files)
	if not files_ids:
		return str('Please add files')
	else: 
		savedFile = ''
		for file_id in files_ids:
			imagefile = flask.request.files[file_id]
			filename = werkzeug.utils.secure_filename(imagefile.filename)
			timestr = time.strftime("%Y%m%d-%H%M%S")
			savedFile = timestr+'_'+filename
			imagefile.save(savedFile)
		#Before prediction
		K.clear_session()
		model= load_model('Model_2019-12-01.h5',compile=False)
		labels=['Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
				'Grape___healthy',
				'Potato___Early_blight', 'Potato___Late_blight',
				'Potato___healthy', 'Tomato___Bacterial_spot',
				'Tomato___Early_blight', 'Tomato___Late_blight',
				'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
				'Tomato___Spider_mites Two-spotted_spider_mite',
				'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
				'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
		lb= LabelBinarizer()
		labels=lb.fit_transform(labels)
		#2nd capture image:
		im_size=150
		img = cv2.imread(savedFile)
		imgResize=cv2.resize(img, (im_size, im_size))
		x=np.array(imgResize)/255.0
		y=np.reshape(x,(1,im_size,im_size,3))

		#4th predict
		prediction =model.predict(y,verbose=1)
		prob=model.predict_proba(y)
		#bacterial_spot
		#5th Match labels code with class name
		indx= np.argmax(prediction ,axis=1) # Replace Y_train to prediction
		label=lb.classes_[indx]
		result=label[0]
		propability=str(int(round(prob[0][indx[0]]*100)))
		# return {
		# 	'propability': propability, 
		# 	'label': result
		# } 
		K.clear_session()
		return result; 
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 33507)), debug=True)