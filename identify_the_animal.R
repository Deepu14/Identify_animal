rm(list = ls())
setwd("/Users/syalamanchi/Desktop/Identify_animal/")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
library(stringr)
library(rlist)
library(tensorflow)
library(keras)

#Keras expects images to be arranged in a directory containing one subdirectory per image class, like
#input/
#  train/
#  class_0/
#  class_0_0.jpg
#class_0_1.jpg
#...
#class_1/
#  class_1_0.jpg
#class_1_1.jpg
#...
#...

#Rename each picture file with their actual animal name 
original_names <- list.files(path = "~/Downloads/DL_Beginner/train/")
abc <-read.csv(file = "~/Downloads/DL_Beginner/train/train.csv")
abc$Image_id<-as.character(abc$Image_id)
abc$Animal<-as.character(abc$Animal)
unique(abc$Animal) #there are 30 species of animals
setdiff(abc$Image_id,original_names) #These image id's are not present in original names i.e. in the test dataset.
length(setdiff(abc$Image_id,original_names)) 
new_names<-NULL
for(i in 1:length(original_names)){
  new_names[i] <- paste(abc$Animal[abc$Image_id==original_names[i]],paste("-",(paste(as.numeric(gsub("\\D","",original_names[i])),"jpg",sep = ".")),sep=""),sep = "")
}
#new_names
setwd("~/Downloads/DL_Beginner/train/")
file.rename(list.files(path = "~/Downloads/DL_Beginner/train/"),paste0(new_names))

####now place different species of animals into different folders
#first create files for each species
for(i in unique(abc$Animal)){
  dir.create(i)
}

#now place all species in their respective folders
names <- list.files(path = "~/Downloads/DL_Beginner/train/")
for(i in names){
  if(endsWith(i,"jpg")){
    file.copy(paste0("~/Downloads/DL_Beginner/train/",i),paste0("~/Downloads/DL_Beginner/train/",strsplit(i,"-")[[1]][1]))
  }
  
}

#Remove the files after moving them to their respective folders
for(i in names){
  if(endsWith(i,"jpg")){
    file.remove(i)
  }
  
}

x<-NULL
for(i in list.dirs()){
  if(i!="."){
    a <-list.files(path = paste0("~/Downloads/DL_Beginner/train/",str_replace(i,"./","")))
    x <- list.append(x,length(a))
  }
}
cat("Number of images are:",sum(x))
#first rename all the files in each folder with name of that folder followed by a number starting from 1:n
#then create a validation directory and send some pictures in each folder to that directory

setwd("~/Downloads/DL_Beginner/train/")

dir.create("~/Downloads/DL_Beginner/validation")
a <- list.files("~/Downloads/DL_Beginner/train/")
for(i in a){
  dir.create(paste0("~/Downloads/DL_Beginner/validation/",i))
}

setwd("~/Downloads/DL_Beginner/train/")
for(j in list.dirs()){
  if(j!="."){
    a <- list.files(path = paste0("~/Downloads/DL_Beginner/train/",str_replace(j,"./","")))
    a <- sample(a)
    for(i in a[1:round(length(a)*0.2)]){
      file.copy(paste0("~/Downloads/DL_Beginner/train/",str_replace(j,"./",""),"/",i),paste0("~/Downloads/DL_Beginner/validation/",str_replace(j,"./","")))  
      file.remove(paste0("~/Downloads/DL_Beginner/train/",str_replace(j,"./",""),"/",i))  
    }
    
  }
}

#data set has been divided into training and validation folders

####### Instantiating a small convnet ########
model <- keras_model_sequential()%>%layer_conv_2d(filters = 32,kernel_size = c(3,3),activation = "relu",input_shape = c(150,150,3))%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 64,kernel_size = c(3,3),activation = "relu")%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 128,kernel_size = c(3,3),activation = "relu")%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 128,kernel_size = c(3,3),activation = "relu")%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512,activation = "relu")%>%
  layer_dense(units = 30,activation = "sigmoid")

summary(model)
model%>%compile(loss = "binary_crossentropy",optimizer = "rmsprop",metrics = c("acc"))

#Data should be formatted into appropriately preprocessed floating-point tensors before being fed into the network. Currently, the data sits on a drive as JPEG 
#files, so the steps for getting it into the network are as follows:
#1. Read the picture files
#2. Decode the JPEG content to RGB grid of pixels.
#3. Rescale the pixel values (between 0 and 255) to the [0,1] interval
#image_data_generator() function turn image files on disk into batches of preprocessed tensors.

datagen <- image_data_generator(rescale = 1/255,rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,shear_range = 0.2,zoom_range = 0.2,
                                horizontal_flip = TRUE)

validation_datagen <- image_data_generator(rescale = 1/255)

###*****Validation data should not be augmented
train_generator <- flow_images_from_directory("~/Downloads/DL_Beginner/train/",datagen,
                                              target_size = c(150,150),batch_size = 32,class_mode = "categorical")
validation_generator <- flow_images_from_directory("~/Downloads/DL_Beginner/validation/",validation_datagen,
                                                   target_size = c(150,150),batch_size = 32,class_mode = "categorical")
#These generators yields batches of 150x150 RGB images (shape(50,150,150,3)) and labels in a two dimensional array. 
#There are 50 samples in each batch (the batch size).
#These generators yeilds these batches indefinitely - it loops endlessly over the images in the target folder.

batch<-generator_next(train_generator)
str(batch)

##Fitting the model using a batch generator

history <- model%>%fit_generator(train_generator,steps_per_epoch = 100,epochs = 60,validation_data = validation_generator,
                                 validation_steps = 70)
#steps_per_epoch = No of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the 
#next epoch. It should typically be equal to the number of samples if the dataset is divided by the batch size.


#Save the model
model%>%save_model_hdf5("animal_classification.h5")
model <- load_model_hdf5(filepath = "animal_classification.h5")
img_path <- "~/Downloads/DL_Beginner/test/Img-50.jpg"
img <- image_load(img_path,target_size = c(150,150))%>%image_to_array()%>%array_reshape(dim = c(1,150,150,3))

dim(img)
#img1 <- k_stack(img)
(preds <- model%>%predict_proba(img))


################################Using a Pretrained Convnet#################################
#There are two ways to use a pretrained network: feature extraction and fine-tuning

#instantiating VGG16 convolutional base
conv_base <- application_vgg16(weights = "imagenet",include_top = FALSE,input_shape = c(150,150,3))
#weight specifies the weight checkpoint from which to initialize the model.
#include_top refers to including the densely connected classifier on top of the network. By default this densely connected classifier 
#corresponds to the 1,000 classes from ImageNet. because we intend to use our own densely connected classifier 
#(with 30 different classes), we don't need to include it.

#input_shape is the shape of the image tensors that we will feed to the network. This argument is optional.

conv_base

#########Feature extraction with data augmentation##########
base_dir <- "~/Downloads/DL_Beginner"
train_dir <- file.path(base_dir,"train")
validation_dir <- file.path(base_dir,"validation")
test_dir <- file.path(base_dir,"test")

model_data_aug <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 30, activation = "sigmoid")
#Before compiling and training the model, it’s very important to freeze the convolutional base. i.e. we are preventing their weights from being updated.
#If we don’t do this, then the representations that were previously learned by the convolutional base will be modified during training.
freeze_weights(conv_base)
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical" 
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)
model_data_aug %>% compile(
  loss = "binary_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)
history <- model_data_aug %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)
model_data_aug%>%save_model_hdf5("pretrained_with_data_aug.h5")
model_data_aug <- load_model_hdf5(filepath = "pretrained_with_data_aug.h5")

#img_path <- "~/Downloads/DL_Beginner/test/unknown/Img-11.jpg"
#img <- image_load(img_path,target_size = c(150,150))%>%image_to_array()%>%array_reshape(dim = c(1,150,150,3))
#img <- img/255
#dim(img)
#preds <- model_data_aug%>%predict(img)

test_datagen1 <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen1,target_size = c(150,150),
  batch_size = 1,
  class_mode = NULL,
  shuffle = FALSE
)


all_preds <- model_data_aug%>%predict_generator(test_generator,steps = 6000,verbose = 1)
all_preds1 <- all_preds

colnames(all_preds1) <- names(train_generator$class_indices)

library(naturalsort)
image_id <- list.files('~/Downloads/DL_Beginner/test/unknown')
image_id <- as.data.frame(image_id)
preds_with_id <- cbind(image_id,all_preds1)
preds_with_id <- preds_with_id[naturalsort(preds_with_id$image_id,decreasing = FALSE),]
write.csv(preds_with_id,file = "predictions.csv",row.names = FALSE)

########################################Fine-tuning##################################
#Here we unfreeze few of the top layers of a frozen model base used for feature extraction and jointly training both the newly
#included part of the model and these top layers. This is called fine-tuning because it slightly adjusts the more abstract 
#representations of the model being reused, in order to make them more relevant for the problem at hand.

#We can only fine tune the top layers of the convolutional base once the classifier on the top has already been trained.
#Steps for fine tuning the network are as follows:
#1 Add your custom network on top of an already-trained base network.
#2 Freeze the base network.
#3 Train the part you added.
#4 Unfreeze some layers in the base network.
#5 Jointly train both these layers and the part you added.

#Here we will fine tune all of the layers from block3_conv1 on.

unfreeze_weights(conv_base,from = 'block3_conv1')

model_data_aug %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)
history <- model_data_aug %>% fit_generator(
  train_generator,
  steps_per_epoch = 60,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

model_data_aug%>%save_model_hdf5("pretrained_with_finetuning.h5")

test_datagen1 <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen1,target_size = c(150,150),
  batch_size = 1,
  class_mode = NULL,
  shuffle = FALSE
)


all_preds <- model_data_aug%>%predict_generator(test_generator,steps = 6000,verbose = 1)
all_preds1 <- all_preds

colnames(all_preds1) <- names(train_generator$class_indices)

library(naturalsort)
image_id <- list.files('~/Downloads/DL_Beginner/test/unknown')
image_id <- as.data.frame(image_id)
preds_with_id <- cbind(image_id,all_preds1)
preds_with_id <- preds_with_id[naturalsort(preds_with_id$image_id,decreasing = FALSE),]
write.csv(preds_with_id,file = "predictions1.csv",row.names = FALSE)




