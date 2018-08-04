rm(list = ls())
setwd("/Users/syalamanchi/Desktop/Identify_animal/")
#devtools::install_github("rstudio/keras")
#library(keras)
#install_keras()
library(stringr)
library(rlist)
library(tensorflow)
library(keras)

#Keras expects images to be arranged in a directory containing one subdirectory per image class, like
input/
  train/
  class_0/
  class_0_0.jpg
  class_0_1.jpg
  ...
class_1/
  class_1_0.jpg
  class_1_1.jpg
  ...
  ...
#Note that this applies even for non-classification tasks. flow_from_directory would still expect a directory that contains a subdirectory with images when class_mode is None.

#Rename each picture file with their actual animal name 
original_names <- list.files(path = "/Users/syalamanchi/Downloads/DL_Beginner/train/")
abc <-read.csv(file = "/Users/syalamanchi/Downloads/DL_Beginner/train/train.csv")
abc$Image_id<-as.character(abc$Image_id)
abc$Animal<-as.character(abc$Animal)
unique(abc$Animal) #there are 30 species of animals
setdiff(abc$Image_id,original_names) #These image id's are not present in original names i.e. in the test dataset.
length(setdiff(abc$Image_id,original_names)) #i.e. 1136 pictures are missing after extracting the folder.
new_names<-NULL
for(i in 1:length(original_names)){
  new_names[i] <- paste(abc$Animal[abc$Image_id==original_names[i]],paste("-",(paste(as.numeric(gsub("\\D","",original_names[i])),"jpg",sep = ".")),sep=""),sep = "")
}
#new_names
setwd("/Users/syalamanchi/Downloads/DL_Beginner/train/")
file.rename(list.files(path = "/Users/syalamanchi/Downloads/DL_Beginner/train/"),paste0(new_names))

######now place different species of animals into different folders
#first create files for each species
for(i in unique(abc$Animal)){
  dir.create(i)
}

#now place all species in their respective folders
names <- list.files(path = "/Users/syalamanchi/Downloads/DL_Beginner/train/")
for(i in names){
  if(endsWith(i,"jpg")){
    file.copy(paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",i),paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",strsplit(i,"-")[[1]][1]))
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
    a <-list.files(path = paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(i,"./","")))
    x <- list.append(x,length(a))
  }
}
cat("Number of images are:",sum(x))
#first rename all the files in each folder with name of that folder followed by a number starting from 1:n
#then create a validation directory and send some pictures in each folder to that directory

setwd("/Users/syalamanchi/Downloads/DL_Beginner/train/")
#for(j in list.dirs()){
#  if(j!="."){
#    a <- list.files(path = paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(j,"./","")))
#    fnames <- paste0(paste0(gsub("\\W","",j),'-'),seq(1:length(a)),".jpg")
#    setwd(paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(j,"./","")))
#    file.rename(list.files(pattern=fixed("antelope")%R%fixed("-")%R%one_or_more(DIGIT)%R%fixed(".jpg")), paste0("antelope-", seq(1:length(a)),".jpg"))
#  }
#}

dir.create("/Users/syalamanchi/Downloads/DL_Beginner/validation")
a <- list.files("/Users/syalamanchi/Downloads/DL_Beginner/train/")
for(i in a){
  dir.create(paste0("/Users/syalamanchi/Downloads/DL_Beginner/validation/",i))
}

setwd("/Users/syalamanchi/Downloads/DL_Beginner/train/")
for(j in list.dirs()){
  if(j!="."){
    a <- list.files(path = paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(j,"./","")))
    a <- sample(a)
    for(i in a[1:round(length(a)*0.2)]){
      file.copy(paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(j,"./",""),"/",i),paste0("/Users/syalamanchi/Downloads/DL_Beginner/validation/",str_replace(j,"./","")))  
      file.remove(paste0("/Users/syalamanchi/Downloads/DL_Beginner/train/",str_replace(j,"./",""),"/",i))  
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
train_generator <- flow_images_from_directory("/Users/syalamanchi/Downloads/DL_Beginner/train/",datagen,
                                              target_size = c(150,150),batch_size = 32,class_mode = "categorical")
validation_generator <- flow_images_from_directory("/Users/syalamanchi/Downloads/DL_Beginner/validation/",validation_datagen,
                                                   target_size = c(150,150),batch_size = 32,class_mode = "categorical")
#These generators yields batches of 150x150 RGB images (shape(50,150,150,3)) and labels in a two dimensional array. 
#There are 50 samples in each batch (the batch size).
#These generators yeilds these batches indefinitely - it loops endlessly over the images in the target folder.

batch<-generator_next(train_generator)
str(batch)
##Fitting the model using a batch generator

#The fit generator function expects as its first argument a generator that will yield batches of inputs and targets indefinitely.
#Because the data is being generated endlessly, the fitting process needs to know how many samples to draw from the generator before declaring an epoch over.
#This is the role of the steps_per_epoch argument: after having drawn steps_per_epoch batches from the generator-after having run for steps_per_epoch
#gradient descent steps, the fitting process will go on to the next epoch. In this case, batches are 50 samples, so it will take 180 batches until you see 
#your target of 9000 samples.


history <- model%>%fit_generator(train_generator,steps_per_epoch = 100,epochs = 60,validation_data = validation_generator,
                                 validation_steps = 70)
#steps_per_epoch = No of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the 
#next epoch. It should typically be equal to the number of samples if the dataset is divided by the batch size.


#Save the model
model%>%save_model_hdf5("animal_classification.h5")

######Setting up a data augmentation configuration via image_data_generator################
datagen <- image_data_generator(rescale = 1/255,rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,shear_range = 0.2,zoom_range = 0.2,
                                horizontal_flip = TRUE,fill_mode = "nearest")

fnames <- list.files("/Users/syalamanchi/Downloads/DL_Beginner/train/antelope/",full.names = TRUE)
img_path <- fnames[[3]]
img <- image_load(img_path,target_size = c(150,150)) #Reads the image and resizes it
img_array <- image_to_array(img) #converts into an array with shape (150,150,3)
img_array <- array_reshape(img_array,c(1,150,150,3)) #reshapes it to (1,150,150,3)
augmentation_generator <- flow_images_from_data(img_array,generator = datagen,batch_size = 1) # Generates batches of randomly transformed images, Loops indefinitely, so you need to break the loop at some point.
#plotting images
op <- par(mfrow=c(2,2),pty="s",mar=c(1,0,1,0))
for(i in 1:4){
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)


################################Using a Pretrained Convnet#################################
#There are two ways to use a pretrained network: feature extraction and fine-tuning
#Feature extraction consists of using the representations learned by the previous network  to extract interesting features from new samples.
#The features are then run through the new classifier , which is trained from scratch.

#In case of convnets, feature extraction consists of taking the convolutional base of a previously trained network, running the new data
#through it, and training a new classifier on top of the output.

#layers that come earlier in the model extract local, highly generic feature maps (such as visual edges, colors and textures), whereas layers
#that are higher up extract more abstract concepts (such as "cat ear" or "dog eye"). So if the new dataset differs a lot from the dataset
#on which the original model was trained , you may be better off using only the first few layers of the model to do feature extraction, rather
#using the entire convolutional base.

#instantiating VGG16 convolutional base
conv_base <- application_vgg16(weights = "imagenet",include_top = FALSE,input_shape = c(150,150,3))
#weight specifies the weight checkpoint from which to initialize the model.
#include_top refers to including the densely connected classifier on top of the network. By default this densely connected classifier 
#corresponds to the 1,000 classes from ImageNet. because we intend to use our own densely connected classifier 
#(with 30 different classes), we don't need to include it.

#input_shape is the shape of the image tensors that you'll feed to the network. This argument is optional.

conv_base

#the final feature map has shape (4,4,512). That's the feature on top of which we will stick a densely connected classifier.
#1. Fast Feature Extraction with out data Augmentation:Here we will run the Convolutional base over our dataset, recording its output  to an array on the disk,
#and then using this data as input to a standalone, densely connected classifier. This solution is cheap to run, because it only requires
#running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline

#Feature extraction consists of taking the convolutional base of a previously trained network, running the new data through it,
#and training a new classifier on top of the output.


#2. Feature Extraction with data Augmentation - slower and expensive, but allows us to use data augmentation during training:
#extending the new_conv base model and running it end to end on the inputs. (Run this only on a GPU)


#######################################Fast Feature Extraction with out data Augmentation#########################################
base_dir <- "~/Downloads/DL_Beginner"
train_dir <- file.path(base_dir,"train")
validation_dir <- file.path(base_dir,"validation")
test_dir <- file.path(base_dir,"test")
datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20
extract_features <- function(directory, sample_count){
  features <- array(0,dim = c(sample_count,4,4,512))
  labels <- array(0,dim = c(sample_count,30))
  generator <- flow_images_from_directory(directory = directory,generator = datagen,target_size = c(150,150),
                                          batch_size = batch_size,class_mode = "categorical")
  i <- 0
  while(TRUE){
    batch <- generator_next(generator) #generator_next retrieves the next item
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)#this extracts interesting features from new samples using representations learned by a previous network.These features are then run through a new classifier, which is trained from scratch.
    #Feature Extraction consists of taking the convolutional base of a previously trained network, running the new data through it, and training a new classifier on top of the output.
    index_range <- ((i*batch_size)+1):((i+1)*batch_size)
    features[index_range, , , ] <- features_batch
    labels[index_range,] <- labels_batch
    i<-i+1
    if(i*batch_size >= sample_count)
      break  #Because generators yield data indefinitely in a loop, you must break after every image has been seen once.
  }
  list(features=features,labels=labels)
 
}


train <- extract_features(train_dir,9400)
validation <- extract_features(validation_dir,2300)
#test <- extract_features(test_dir,6000)

#The extracted features are currently of shape (samples,4,4,512). We will feed them to a densely connected classifier, 
#so first you must flatten them to (samples,8192):
reshape_features <- function(features){
  array_reshape(features,dim = c(nrow(features),4*4*512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
#test$features <- reshape_features(test$features)

######Defining and training a densely connected classifier######
model_pretrained <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu",input_shape = 4*4*512) %>%layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%layer_dropout(rate = 0.5) %>%
  layer_dense(units = 30,activation = "sigmoid")
model_pretrained %>% compile(optimizer = "rmsprop",loss = "binary_crossentropy",metrics = c("accuracy"))
history <- model_pretrained %>% fit(train$features,train$labels,epochs = 30, batch_size = 20,
                                    validation_data = list(validation$features,validation$labels))
model_pretrained%>%save_model_hdf5("model_pretrained.h5")
#model_pretrained <- load_model_hdf5(filepath = "model_pretrained.h5")

img_path <- "~/Downloads/DL_Beginner/test/Img-10.jpg"
img <- image_load(img_path,target_size = c(150,150))%>%image_to_array()%>%array_reshape(dim = c(1,150,150,3))%>%imagenet_preprocess_input()
#img <- img/255
dim(img)
#img <- k_expand_dims(img,axis = 0)
#img1 <- k_stack(img)
preds <- model_pretrained%>%predict_classes(img)
preds







#test_datagen <- image_data_generator(rescale = 1/255)

#test_generator <- flow_images_from_directory(directory = "~/Downloads/DL_Beginner/test/",generator = test_datagen,target_size = c(150,150),
#                                        batch_size = batch_size,class_mode = "categorical")
#preds <- predict_generator(model_pretrained,test_generator,steps = )
