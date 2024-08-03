if __name__=="__main__":
	import streamlit as st
	import os
	import matplotlib.pyplot as plt
	import numpy as np
	from PIL import Image
	from keras.preprocessing.image import ImageDataGenerator
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	
	#1. Create a streamlit title widget, this will be shown first
	st.title("Plant Seeds Identificator")
			
	upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
	c1, c2= st.columns(2)
	if upload is not None:
		# Open the image with PIL and reshape it to 224,224,3
		image= Image.open(upload)
		
		# Instantiate ImageDataGenerator to perform pre-processing on the loaded image
		image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
		image_resize = 224 # the target size of VGG
		
		# reshape it to 224,224,3 and display
		image_resized = image.resize((image_resize, image_resize))
		c1.image(image_resized)
		
		image_resized = np.array(image_resized) # as numpy array
		
		# use image_generator to transform the image_transformed
		x = np.expand_dims(image_resized, axis=0)
		img_transformed = image_generator.flow(x)
		
		c1.header('Input Image')
		#c1.write(y.shape)
		
		file_name = "saved_models/model.keras"
		if os.path.isfile(file_name):
			pass
		else:
			raise Exception("model not found!")
		
		#load the model
		model = load_model(file_name)
		
		# Evaluate on test_generator
		eval = model.predict(img_transformed)
		# transforms smt like [0.33,0.67] into [0,1]
		pred = np.argmax(eval,axis=-1) #(eval_vgg > 0.5).astype("int32")
			
		class_names = {0: 'apple_6',
 						1: 'apple_braeburn_1',
 						2: 'apple_crimson_snow_1',
 						3: 'apple_golden_1',
 						4: 'apple_golden_2',
 						5: 'apple_golden_3',
 						6: 'apple_granny_smith_1',
 						7: 'apple_hit_1',
 						8: 'apple_pink_lady_1',
 						9: 'apple_red_1',
 						10: 'apple_red_2',
 						11: 'apple_red_3',
 						12: 'apple_red_delicios_1',
 						13: 'apple_red_yellow_1',
 						14: 'apple_rotten_1',
 						15: 'cabbage_white_1',
 						16: 'carrot_1',
 						17: 'cucumber_1',
 						18: 'cucumber_3',
 						19: 'eggplant_long_1',
 						20: 'pear_1',
 						21: 'pear_3',
 						22: 'zucchini_1',
 						23: 'zucchini_dark_1'}
		
		print("predicted", int(pred),)
		ind = int(pred)
		prediction = class_names[ind]
			
		c2.header('Output')
		c2.subheader('The seed you have is :')
		c2.write(prediction)
