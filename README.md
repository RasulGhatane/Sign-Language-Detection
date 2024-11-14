[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2dHb5_fR)
# Group Assignment: [Speak UP!]
## Overview
Welcome to the group project repository for **[Speak UP!]**. This repository is where our team will collaborate, contribute code, documentation, and complete the assignment. Each member is expected to contribute actively to ensure the success of the project.
---
## Team Members
- **Member 1**: Aayushma Thapa Magar - Leader
- **Member 2**: Rasul Ghatane - Researcher
- **Member 3**: Samyak Raj Shakya - Frontend Developer
- **Member 4**: Aaditya Chand - Model Trainer
---

## Project Description
### [Speak UP!]
Speak UP! is an innovative program designed to empower individuals with verbal disabilities by helping them communicate effectively with others. The core idea is to bridge the gap between sign language users and non-signers through real-time sign language detection. By leveraging the power of machine learning, Speak UP! can recognize specific hand gestures and signs, interpret them, and translate them into spoken or written language, enabling smoother interactions.

The program likely uses a trained machine learning model, such as the one saved in keras_model.h5, to classify various sign language gestures based on visual input. This model could be trained on a dataset of labeled sign language images or videos to accurately recognize and differentiate between a variety of hand signs. The labels in labels.txt may correspond to specific signs or gestures, acting as a reference for the model to map detected gestures to corresponding words or phrases.

Here's a potential workflow:

Data Collection: To train a machine learning model for gesture recognition, Speak UP! might gather a dataset of images or video frames capturing different sign language gestures. These data samples are then labeled to help the model understand the relationship between a gesture and its meaning.

Model Training: Using the collected data, a machine learning model is trained, likely with a framework like Keras. This model learns to recognize patterns in the gestures, allowing it to differentiate between signs. The training process involves feeding the model labeled examples repeatedly until it can accurately classify each gesture.

Real-time Detection: Once trained, the model can be used in real-time. As the user performs gestures, the program interprets these signs by comparing them against the learned patterns and provides instant feedback in the form of text or synthesized speech.

User Interface: The program could have a user-friendly interface, perhaps a web-based application with a frontend (the index.html file in the templates directory). Users might simply need a camera-enabled device to capture their gestures, and the program would display the recognized signs on screen or convert them to audio.

Overall, Speak UP! aims to make communication easier and more inclusive, helping individuals with verbal disabilities to interact with others independently. By harnessing machine learning for sign language detection, it can potentially make a positive impact in various real-world scenarios, like customer service, healthcare, and everyday social interactions.



### [How to run this project]
- Step 1: Install the reqired libraries. 
          On Terminal Run:  pip install -r requirements.txt         
- Step 2: Run input.py to collect and train the data. Input the name and number of sequences for the data. 
- Step 3: Run app.py to run the program. Navigate to http://127.0.0.1:5000 on your browser to run the code. 
    
    
  
---
## Repository Structure
```plaintext
project_directory/
├── templates/
│   └── index.html
├── app.py
├── input.py
├── keras_model.h5
├── labels.txt
└── requirements.txt


│   └── Member4
│   └── Member5
│
├── README.md               # This README file
└── requirements.txt        # Python dependencies
