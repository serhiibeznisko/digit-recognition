## Project for handwritten digits recognition with Keras and Django frameworks. ##

The Neural Network is trained by **Keras** neural network API which runs on top of **TensorFlow** framework.
The Neural Network built for this project is **Convolutional Neural Network** that consists of 10 layers of 6 different types that are combined together, making training as efficient as possible.
To train the model it uses popular [mnist](https://keras.io/api/datasets/) dataset. Overall, there are 60,000 samples for training and 10,000 for testing.
After neural network on the dataset is trained, it is saved in the file mnist.h5 for later use. Later, to let the user test the model,
a small **Django Web Application** makes it possible for users to test and predict their own handwritten digits or continue to train the model further in case it still gives the wrong results.

### To run the project you need to have Python3 installed in your system. ###
In the root folder of the project run:

```
python3 -m pip install -r requirements.txt
cd src
python3 train_model.py (in case you want to train the model again)
python3 manage.py runserver localhost:8001 --nothreading --noreload
```

Now you can go to http://localhost:8001 and check the neural network accuracy.

Structure of the project:

```
├── README.txt
├── requirements.txt (project dependencies)
└── src
    ├── Makefile
    ├── __init__.py
    ├── app
    │   ├── __init__.py
    │   ├── settings.py (server settings)
    │   ├── urls.py (server routes)
    │   ├── views.py (server endpoints)
    │   └── wsgi.py
    ├── manage.py
    ├── mnist.h5 (your trained model data)
    ├── templates
    │   └── home.html (html/css/js for frontend)
    └── train_model.py (script to train the model)
```

### Sources used to create the project: ###
https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/  
https://docs.djangoproject.com/en/3.0/  
https://keras.io/api/  
https://stackoverflow.com (forever in our hearts)  

