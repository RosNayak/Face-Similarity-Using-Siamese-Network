# Face-Similarity-Using-Siamese-Network

## Dataset
The dataset for this project was obtained from [this](https://github.com/Ruturaj123/Face-Similarity-Siamese-Network) github repository by Ruturaj.
The train dataset had 10 faces images of 10 different persons, while the test dataset had 30 images of 3 different persons. The dataset was created using random sampling for the training and testing purpose. The training was done on 2220 images with validation size of 20% and tested on 60 images.

## Contrastive Loss
<img width="750" alt="image" src="https://user-images.githubusercontent.com/45042726/165827500-cebcfcb1-d8dd-4213-bf8b-5a7977fcacb9.png">

### Objective
The objective of the project is to create a deep learning network which identifies if a pair of images belong to the same person or not using Siamese network and contrastive loss.

### Results
<img width="500" alt="image" src="https://user-images.githubusercontent.com/45042726/165822639-301e7fa4-a183-4d37-a636-4b1833fa9c90.png">

## Triplet Loss
<img width="750" alt="image" src="https://user-images.githubusercontent.com/45042726/166147498-fb894dea-0d0d-4b72-b96a-afd695faf010.png">

### Objective
The objective of the project is to create a deep learning network which identifies images closest to the input face image using Siamese network and triplet loss.

### Results
<img width="500" alt="image" src="https://user-images.githubusercontent.com/45042726/166146708-d3c6a4e3-b51e-4668-9f87-6dd70dfb1092.png">

<img width="526" alt="image" src="https://user-images.githubusercontent.com/45042726/166147650-bd426e1e-0de8-4987-85ef-10a53c6db889.png">

Using Principle Component Analysis the embedding vector of dimension 64 was reduced to size of 2 so that we can visualize the distribution of the image embeddings in the vector space. The image below show exactly that.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/45042726/166146776-cf911b6a-bc7a-4f67-a090-052f1675c794.png">

We see that the test set had images of three different persons and the embeddings of the images person 0 and 1 are close to each other and form a cluster. Whereas 7 out of 10 images of person 2 are forming a cluster while the other 3 donot belong to that cluster.

Also displaying the top 5 closest images to the first image in the row shown in the image below.

#### Person 0
<img width="604" alt="image" src="https://user-images.githubusercontent.com/45042726/166147277-2817f01f-9946-4cb3-b352-54e5b7f6c866.png">

#### Person 1
<img width="604" alt="image" src="https://user-images.githubusercontent.com/45042726/166147237-314169a2-056c-4544-ba8c-7c8032a87cbd.png">

#### Person 2
<img width="604" alt="image" src="https://user-images.githubusercontent.com/45042726/166147354-e73d329b-9dac-4482-b6bb-96360ad76c0d.png">
The first 3 images are those that are closest to the embeddings of person 1 mostly the reason being that the emotions and face orientations are same across all the images.
