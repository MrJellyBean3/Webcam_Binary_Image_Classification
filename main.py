import time
import os
print(os.path.isfile("auto_web5/U/186_U_1.jpeg"))#C:/CAMERA_AI/
#sys.path.append()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pygame.camera
import cv2
import scipy
import keyboard
import math


initialize_data=True
view_conv=False
train_path_set="auto_web5"
validation_path_set="auto_web5_v"#"web5_1bad"
validation2_path_set="auto_web5_v"#"web5_1bad"
save_dir="auto_web5"


def make_circle_pos(pos, point_count, radius):
    grab_pos=[pos[0],pos[1]]
    radial_pos_array=[]#horizontal then vertical
    radius=50
    point_count=20
    angle_inc=(360/point_count)
    for place in range(point_count):
        radial_pos_array.append([grab_pos[0]+int(radius*(math.cos(angle_inc*place))),grab_pos[1]+int(radius*(math.sin(angle_inc*place)))])
    radius=radius/2
    point_count=int(point_count/2)
    angle_inc=(360/point_count)
    for place in range(point_count):
        radial_pos_array.append([grab_pos[0]+int(radius*(math.cos(angle_inc*place))),grab_pos[1]+int(radius*(math.sin(angle_inc*place)))])
    radius=8
    point_count=5
    angle_inc=(360/point_count)
    for place in range(point_count):
        radial_pos_array.append([grab_pos[0]+int(radius*(math.cos(angle_inc*place))),grab_pos[1]+int(radius*(math.sin(angle_inc*place)))])
    radius=1
    point_count=3
    angle_inc=(360/point_count)
    for place in range(point_count):
        radial_pos_array.append([grab_pos[0]+int(radius*(math.cos(angle_inc*place))),grab_pos[1]+int(radius*(math.sin(angle_inc*place)))])
    return(radial_pos_array)

def cropper_ft(frame, im_size, pos):
    size_og=im_size
    im_size[0]=int(im_size[0]/2)
    im_size[1]=int(im_size[1]/2)
    if (im_size[0]*2>frame.shape[1]) or (im_size[1]*2>frame.shape[0]):
        run=True
        bad_crop=False
        im_size[0]=frame.shape[1]/2
        im_size[0]=frame.shape[0]/2
    else:
        run=True
    xmin=pos[0]-im_size[0]
    xmax=pos[0]+im_size[0]
    ymin=pos[1]-im_size[1]
    ymax=pos[1]+im_size[1]
    bad_crop=True
    timer=time.time()
    
    while run:
        if xmin<0:
            print(1)
            pos[0]+=1
        elif ymin<0:
            print(2)
            pos[1]+=1
        elif xmax>((frame.shape)[1]):
            print(3)
            pos[0]-=1
        elif ymax>(frame.shape)[0]:
            print(4)
            pos[1]-=1
        else:
            run=False
            bad_crop=False
        if ((time.time()-timer)>5):
            bad_crop=True
            print("bad crop")
            run=False
        #print(pos)
    xmin=pos[0]-im_size[0]
    xmax=pos[0]+im_size[0]
    ymin=pos[1]-im_size[1]
    ymax=pos[1]+im_size[1]
    frame2=frame[ymin:ymax,xmin:xmax]
    im_size=size_og
    return(frame2)

def make_train_images(frame_array,  size_train, pos,unlocked_bool,in_save_directory):
    initial_size=0
    initial_size=[size_train[0], size_train[1]]
    print("making training images starting now")
    if in_save_directory[-1]!="/":
        in_save_directory+="/"
    counter=0
    variance_translate=50
    variance_zoom=0.25
    unique_id=str(int(np.random.rand(1)*1000))
    if unlocked_bool:
        save_path=in_save_directory+unique_id+"_U_1.jpeg"
    else:
        save_path=in_save_directory+unique_id+"_L_1.jpeg"
    no_unique_counter=0
    while os.path.isfile(save_path)and(no_unique_counter<1000):
        no_unique_counter+=1
        unique_id=str(int(np.random.rand(1)*1000))
        if unlocked_bool:
            save_path=in_save_directory+unique_id+"_U_1.jpeg"
        else:
            save_path=in_save_directory+unique_id+"_L_1.jpeg"



    rand_translation=(np.random.rand(16,2)*variance_translate/2+variance_translate/2)
    for frame_ind in range(len(frame_array)):
        for row_ind in range(len(rand_translation)):
                ind=row_ind%2
                if ind==0: 
                    rand_zoom=(np.random.rand(2))
                    rand_zoom=[rand_zoom[0]*variance_zoom/2+1-variance_zoom/3,1+variance_zoom/3-rand_zoom[1]*variance_zoom] 
                x_pn=(1 if np.random.random() < 0.5 else -1)
                y_pn=(1 if np.random.random() < 0.5 else -1)
                print([rand_translation[row_ind][0]*x_pn,rand_translation[row_ind][1]*y_pn], rand_zoom[ind])
                crop_pos=[int(pos[0]+rand_translation[row_ind][0]*x_pn),int(pos[1]+rand_translation[row_ind][1]*y_pn)]
                #size_train=initial_size
                size_train[0]=int(initial_size[0]*rand_zoom[ind])
                size_train[1]=int(initial_size[1]*rand_zoom[ind])
                frame=frame_array[frame_ind]
                print(size_train)
                data_frame=cropper_ft(frame, size_train, crop_pos)
                data_frame=cv2.resize(data_frame,dsize=(model_image_size[1],model_image_size[0]),interpolation=cv2.INTER_CUBIC)
                if unlocked_bool:
                    general_save_path=in_save_directory+unique_id+"_U_"
                else:
                    general_save_path=in_save_directory+unique_id+"_L_"
                counter+=1
                data_frame=cv2.cvtColor(data_frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(general_save_path+str(counter)+".jpeg",data_frame)





#dir_5="circle5"

#dir_5="web5_1bad"
#dir_5V="auto_web5_V"
#dir_5V="web5_3"
#dir_5V="circle_val"

#dir_5V="web5R_V"
b_size=20
#model_image_size=[270, 480]
model_image_size=[270, 345]
into_model_height=model_image_size[0]
into_model_width=model_image_size[1]
color_set="rgb"








if color_set=="rgb":  #the model and checkpoint
  shape_set=3
else:
  shape_set=1
into_model_height=int(into_model_height)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (4,4), activation='relu', input_shape=(into_model_height, into_model_width, shape_set)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(24, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, 'sigmoid')
])
model.summary()
model.compile(loss='MSE',
              optimizer='RMSprop',
              #optimizer=RMSprop(learning_rate=0.0005),
              metrics=['accuracy'])

checkpoint_filepath = 'checkpoints/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)








if initialize_data:
#train generator 
    train_path=train_path_set#path for datagen                      THIS IS ALL THE DATA FOR TRAINING
    print(train_path)
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        #brightness=0.2,
        #channel_shift_intensity=0.2,
        horizontal_flip=True,
        rotation_range=20,
        channel_shift_range=40,
        #zca_whitening=True,
        #zca_epsilon=200,
        brightness_range=[0.80,1.20],
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=[0.9,1.1],#[0.6,0.9],
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
            train_path,  # This is the source directory for training images
            target_size=(into_model_height, into_model_width),  # All images will be resized to 300x300
            #color_mode='grayscale',
            class_mode='binary',
            color_mode=color_set,
            batch_size=b_size)
    print(len(train_generator.labels))
    print(len(train_generator))

    #Validation Generator 1
    val_path2=validation_path_set
    val_datagen=ImageDataGenerator(rescale=1/255,brightness_range=[0.9,1.1],zoom_range=[0.9,1.1],width_shift_range=0.25,
        height_shift_range=0.25)
    validation_generator= val_datagen.flow_from_directory(val_path2,
            target_size=(into_model_height, into_model_width),  # All images will be resized to 300x300
            class_mode='binary',
            color_mode=color_set)
        
    print(len(validation_generator.labels))
    print(len(validation_generator))


    #validation generator 2
    dir_5V_2=validation2_path_set
    val_path2=dir_5V_2  
    val_datagen2=ImageDataGenerator(rescale=1/255,brightness_range=[0.9,1.1],zoom_range=[0.9,1.1])
    validation_generator2= val_datagen2.flow_from_directory(val_path2,
            target_size=(into_model_height, into_model_width),  # All images will be resized to 300x300
            class_mode='binary',
            color_mode=color_set)
        
    print(len(validation_generator2.labels))
    print(len(validation_generator2))

    f_view=plt.figure()
    f_view.set_figheight(25)
    f_view.set_figwidth(25)
    figw=5
    figh=5
    train_view=train_generator[0][0][0]
    im_train=train_view
    print(im_train.shape)
    plt.imshow(np.squeeze(im_train[None]))
    plt.show()
    #time.sleep(3)
    #plt.close()







conv2d_array=[]

if view_conv:
    ind=12
    X=validation_generator[0][0][ind]
    V=validation_generator[0][0][ind]
    X = X.reshape((1,) + X.shape)
    print(X.shape)
    figureh=5
    figurew=4
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    successive_feature_maps = visualization_model.predict(X)
    layer_names = [layer.name for layer in model.layers]
    i=0
    for name, feature_map in zip(layer_names, successive_feature_maps):
        i=i+1
        print(name)
        if name=="average_pooling2d_2" or i==5:
            n_features=16
            for i in range(n_features):
              print(i, 'i')
              x  = feature_map[0, :, :, i]
              x -= x.mean()
              x /= x.std ()
              x *=  64
              x += 128
              x  = np.clip(x, 0, 255).astype('uint8')
              conv2d_array.append(x)
    for loopval in range(0, n_features):
        if ((loopval)<len(conv2d_array)):
             plt.subplot(figureh,figurew,loopval+1)
             plt.imshow(conv2d_array[loopval],cmap=plt.get_cmap('gray'))
    plt.subplot(figureh,figurew,loopval+2)
    plt.imshow(V)#,cmap=plt.get_cmap('gray'))
    plt.pause(50)
    plt.close()









if os.path.isfile("checkpoints/checkpoint"):
    model.load_weights("checkpoints/checkpoint")
    model.save_weights("checkpoints/backupcheckpoint")
    print("train")
    #model.evaluate(train_generator)
    print("validation")
    #model.evaluate(validation_generator)
    print("validation2")
    #model.evaluate(validation_generator2)
print("do you want to train Y/N")
txt=input()
if txt=='Y' or txt=='y':
    training=True
    txt=input("number of epochs")
else:
    training=False

#if training:
    
    #for index_train in range(b_size):
     # plt.subplot(figh,figw,index_train+1)
      #plt.imshow(np.squeeze(train_view[index_train]),cmap=plt.get_cmap('gray'))
      #plt.close()
while training:
    history = model.fit(
      train_generator,  
      epochs=int(txt),
      batch_size=b_size,
      verbose=1,
      validation_data=validation_generator,
      callbacks=[model_checkpoint_callback]
      )
    print(model.evaluate(train_generator))
    print(model.evaluate(validation_generator))
    print("do you want to train more Y/N")
    txt=input()
    if txt=='Y' or txt=='y':
        training=True
        txt=input("number of epochs")
    else:
        training=False
        txt=input("revert to last checkpoint Y/N ")
        if txt=='Y' or txt=='y':
            model.load_weights("checkpoints/checkpoint")
            print("train")
            model.evaluate(train_generator)
            print("validation")
            model.evaluate(validation_generator)
            print("do you want to train more Y/N")
            txt=input()
            if txt=='Y' or txt=='y':
                training=True
                txt=input("number of epochs")

#from cv2 import cvtColor
bgr_im=cv2.imread("auto_web5_v/U/401_U_1.jpeg")
print(bgr_im)
image_from_file=[]
for i in range(1,15):
    bgr_im=cv2.imread("auto_web5_v/U/401_U_"+str(i)+".jpeg")
    temp_im_2=bgr_im[:, :, [2, 1, 0]]/255.0
    image_from_file.append(temp_im_2[None])
print(model.predict(validation_generator[0][0]))
b_test=[]
for i in range(0,14):
    b_test.append(model.predict(image_from_file[i]))
print(b_test)
print("do you want to start the camera")
txt=input("Y/N")
if txt=='Y' or txt=='y':
    runval=True
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)#video capture starts
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    strt_time=time.time()

    ret, frame = vid.read()
    
    frame_shape=frame.shape
    pos=[int(frame_shape[1]/2),int(frame_shape[0]/2)]
    pos=[950,470]
    print(pos)
    zoom=0.3
    size=[int(frame_shape[1]*zoom*0.7),int(frame_shape[0]*zoom)]
    frame=cropper_ft(frame, size, pos)
    frame=cv2.resize(frame,dsize=(model_image_size[1],model_image_size[0]),interpolation=cv2.INTER_CUBIC)#dsize is height x width

    
    fig, ax1= plt.subplots()
    img=ax1.imshow(frame, cmap=plt.get_cmap('gray'))
    plt.ion()

else:
    runval=False
i=0
get_frames_l=False
get_frames_u=False
conv_dipsplay=False
cind=0
c_count=0
c_pressed=False
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
rect_count=0
take_rect_frames=False
prediction=np.zeros([1,1])
prediction[0][0]=0.420

pred_start=time.time()
while runval:
        i+=1
        prev_time=time.time()
        ret, frame = vid.read()
        frame_time=(time.time()-prev_time)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_shape=frame.shape
        size=[int(frame_shape[1]*zoom*0.7),int(frame_shape[0]*zoom)]#define frame




        if (keyboard.is_pressed("0")): #getting training images off of input
            frame_count=0
            frame_array=[0,0,0]
            get_frames_l=True
        elif get_frames_l and frame_count<3:
            frame_array[frame_count]=frame
            frame_count+=1
            if frame_count==3: 
                get_frames_l=False
                print(pos)
                make_train_images(frame_array, size, pos, False, save_dir+"/L")
        if (keyboard.is_pressed("1")):
            frame_count=0
            frame_array=[0,0,0]
            get_frames_u=True
        elif get_frames_u and frame_count<3:
            frame_array[frame_count]=frame
            frame_count+=1
            if frame_count==3:
                get_frames_u=False
                print(pos)
                make_train_images(frame_array, size, pos, True, save_dir+"/U")






        
        if (keyboard.is_pressed("left")):#manipulate camera direction
                    pos[0]-=10
                    print(pos)
        if (keyboard.is_pressed("right")):
                    pos[0]+=10
                    print(pos)
        if (keyboard.is_pressed("up")):
                    pos[1]-=10
                    print(pos)
        if (keyboard.is_pressed("down")):
                    pos[1]+=10
                    print(pos)
        if (keyboard.is_pressed("=")):
                    if zoom>0.01:
                        zoom-=0.01
                        print(zoom)
        if (keyboard.is_pressed("-")):
                    if zoom<1:
                        zoom+=0.01
                        print(zoom)
        if (keyboard.is_pressed("escape")):
                    runval=False
        prev_time=time.time()
        frame=cropper_ft(frame, size, pos)
        frame=cv2.resize(frame,dsize=(model_image_size[1],model_image_size[0]),interpolation=cv2.INTER_CUBIC)#dsize is height x width
        crop_resize_time=time.time()-prev_time



        c_prev=c_pressed#convolution code
        if keyboard.is_pressed("c"):
            c_pressed=True
        else:
            c_pressed=False
        if c_prev!=c_pressed:
            c_count+=1
            if c_count>1:
                c_count=0
                if conv_dipsplay:
                    conv_dipsplay=False
                else:
                    conv_dipsplay=True
        if conv_dipsplay:
            X=frame
            X = X.reshape((1,) + X.shape)
            print(X.shape)
            successive_feature_maps = visualization_model.predict(X, verbose=0)
            layer_names = [layer.name for layer in model.layers]
            i=0
            for name, feature_map in zip(layer_names, successive_feature_maps):
                i=i+1
                if i==5:
                    n_features=16
                      #print(i, 'i')
                    x  = feature_map[0, :, :, cind]
                    x -= x.mean()
                    x /= x.std ()
                    x *=  64
                    x += 128
                    x  = np.clip(x, 0, 255).astype('uint8')
                    
                    rows=len(x)
                    cols=len(x[0])
                    rgb_x=np.zeros([rows,cols,3])
                    for row_i in range(rows):
                        for col_i in range(cols):
                            rgb_x[row_i][col_i][0]=x[row_i][col_i]/255
                            rgb_x[row_i][col_i][1]=x[row_i][col_i]/255
                            rgb_x[row_i][col_i][2]=x[row_i][col_i]/255
                    x=x/255.0
                    print(x)
        if keyboard.is_pressed(","):#change conv display
            cind-=1
            print(cind)
            if cind<0:
                cind=0
        if keyboard.is_pressed("."):
            print(cind)
            cind+=1
            if cind>15:
                cind=15
        if conv_dipsplay:
            img.set_data(rgb_x)
        else:
            prev_time=time.time()
            img.set_data(frame),
        set_time=time.time()-prev_time
        prev_time=time.time()
        if not(keyboard.is_pressed("g")):
            if ((time.time()-pred_start)>0.5):
                pred_start=time.time()
                prediction=model.predict(frame[None]/255.0, verbose=0)#model prediction
        else:
            prediction[0][0]=0.420
        pred_time=time.time()-prev_time
        title_string=str('{0:.2f}'.format(1/(time.time()-strt_time)))+" fps  "+str('{0:.1f}'.format(prediction[0][0]*100))+" %chance unlocked"+"   "+str('{0:.2f}'.format(pred_time))+"    "+str('{0:.2f}'.format(set_time))+"   "+str('{0:.2f}'.format(crop_resize_time))+"    "+str('{0:.2f}'.format(frame_time))
        ax1.set_title(title_string, fontsize=30)#set title




        plt.pause(0.01)
        strt_time=time.time()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (keyboard.is_pressed("i") or keyboard.is_pressed("n")): #circle unlocked data take
            if keyboard.is_pressed("i"):
                unlocked_grab=True
                print("i")
            else:
                print("n")
                unlocked_grab=False
            pos_backup=[pos[0],pos[1]]
            take_rect_frames=True
            rect_count=0
        if take_rect_frames and not(keyboard.is_pressed("2") or keyboard.is_pressed("9")): 
            if rect_count==0:
                pos_array=make_circle_pos(pos, 20,50)#position , points in outer ring, radius in pixels
                len_pos=len(pos_array)
                unique_file_count=1
            pos=pos_array[rect_count]
            rect_count+=1
            if unlocked_grab:
                dir_i="circle5/U/"
                f_name=str(rect_count)+"_U"+str(unique_file_count)+".jpeg"
            else:
                dir_i="circle5/L/"
                f_name=str(rect_count)+"_L"+str(unique_file_count)+".jpeg"

            while os.path.exists(dir_i+f_name):
                unique_file_count+=1
                f_name=str(rect_count)+"_L"+str(unique_file_count)+".jpeg"
            cv2.imwrite(dir_i+f_name, frame)
            if rect_count>=len_pos:
                take_rect_frames=False
                pos=[pos_backup[0],pos_backup[1]]




        if (keyboard.is_pressed("l")):#manual lock and unlocked get data
                    image_name=str(i)+"_locked.jpeg"
                    cv2.imwrite("web5_1/L/"+image_name,frame)
        if (keyboard.is_pressed("u")):
                    image_name=str(i)+"_unlocked.jpeg"
                    cv2.imwrite("web5_1/U/"+image_name,frame)
        time.sleep(0.040)
plt.close()




