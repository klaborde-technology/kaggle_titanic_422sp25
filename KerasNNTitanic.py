import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
#keras
# Import necessary modules
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.saving import load_model
#load the dataset,
df = pd.read_csv("survivor.csv")
print(f"data:\n{df.describe()}")
print(df.columns)

df["age_was_missing"] = df["age_was_missing"].astype(int)
#set predictors and targets
predictors = df.drop(["survived"], axis = 1).values
from keras.utils import to_categorical
target = to_categorical(df['survived'])
#setup input layer
from keras.layers import Input
ncols = predictors.shape[1]
inputs = Input(shape=(ncols,))

#build a sequential model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(inputs)
model.add(Dense(32, activation="relu") )
model.add(Dense(2,activation="softmax" ))
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(predictors, target, epochs=100)
model.summary()

# save the model
model.save("survivor.keras")
model.save("survivor.h5")

# save the model
my_model = load_model("survivor.keras")
#view structure
my_model.summary()

#preprocess the test data
df = pd.read_csv("survivor_test.csv")
print(f"data:\n{df.describe()}")
print(df.head())
#observe True False, convert to int
df["age_was_missing"] = df["age_was_missing"].astype(int)
#set predictors
predictors = df.drop(["survived"], axis = 1).values

#make predictions
pred = my_model.predict(predictors)
# print the probabilitities of survival
print(f"prob survived:{pred[:,1]}")