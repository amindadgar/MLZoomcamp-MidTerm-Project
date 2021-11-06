# MLZoomcamp-MidTerm-Project
This Repository Contains MLZoomcamp Midterm Project Files.

## Dataset
Dataset is taken from https://www.kaggle.com/alakaaay/diabetes-uci-dataset <br>
### What is the dataset used for?
Given dataset has been collected using direct questionnaires from the patients of Sylhet Diabetes
Hospital in Sylhet, Bangladesh, and approved by a doctor. As we train a model for the project, We are predicting a person going to have diabetes or not. With attributes that the dataset have, we can predict how much a person is going to have diabetes or not with a number between 0 to 1, Actually with multiplying it with 100, It would give the percentage of a person having diabetes in future. 

### Modifications:

There are some modification to it to explore more options in machine learning <br> 
- Binary category was added to gender in order to use One-hot encoding (dummy variable) option in training.

### Dataset Content 
Content
Attribute information:
1) Age: 20-65
2) Sex: Male/Female
3) Polyuria: Yes/No
4) Polydipsia: Yes/No
5) sudden weight loss: Yes/No
6) weakness: Yes/No
7) Polyphagia: Yes/No
8) Genital thrush: Yes/No
9) visual blurring: Yes/No
10) Itching: Yes/No
11) Irritability: Yes/No
12) delayed healing: Yes/No
13) partial paresis: Yes/No
14) muscle stiffness: Yes/No
15) Alopecia: Yes/No
16) Obesity: Yes/No
17) Class: Positive/Negative

## Test the project
The project is now deployed on heroku cloud servers and to test it just run the file [test.ipnyb](https://github.com/amindadgar/MLZoomcamp-MidTerm-Project/blob/main/test_file.ipynb). Or you can just send json data into the link: https://diabetes-prediction-server.herokuapp.com/predict and get the results.

## To deploy it by yourself
To deploy this project on your local computer or your server just follow the instructions below:
- First Install Docker on your system
- Install pipenv on your system.
- ```pipenv install requirments.txt```
- ```docker built -t diabetes-prediction .```
- ```docker run -it -p 9696:9696 diabetes-prediction:latest```
