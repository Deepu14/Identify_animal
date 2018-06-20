# Identify_animal
Problem Statement:

Wildlife images captured in a field represent a challenging task while classifying animals since they appear with a different pose, cluttered background, different light and climate conditions, different viewpoints, and occlusions. Additionally, animals of different classes look similar. All these challenges necessitate an efficient algorithm for classification.

In this challenge, you will be given 19,000 images of 30 different animal species. Given the image of the animal, your task is to predict the probability for every animal class. The animal class with the highest probability means that the image belongs to that animal class.

Data Description:

You’re given two types of files (CSV and Images) to download. The train data consists of 13,000 images and the test data consists of 6,000 images of 30 different species of animals. The image ID and the corresponding animal name are stored in .csv format, while the image files are sorted into separate train and test image folders. Data in the .csv file is in the following format:

Variable

Description

Image_id

Image name

Animal

Name of the Animal

There are 30 different species of animals in the dataset.

Download the data from https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.torrent or
https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.zip
