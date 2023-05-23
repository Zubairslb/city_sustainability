# city_sustainability
Bootcamp_project

### installation instruction

pyenv virtualenv city_sustainability

...........................................


pyenv local city_sustainability

............................................

pip install -r requirments.txt

............................................
Install city_sustainabilty package in edit mode

pip install -e .

..............................................

### Package for Model
~~~~~

from city_sustainability.models import unet_model

### Create the model:-)
model = unet_model.build_model(input_shape=(a, b, 3), num_classes=c)

### Compile the model:-)
unet_model.compile_model(model)

### Train the model:-)
unet_model.train_model(model, x_train , y_train  , epochs=1, batch_size=512, validation_split=0.2)

### Evaluate the model:-)
evaluate_model(model, x_test, y_test)

### Make predictions:-)
predictions = predict(model, x_test)

~~~~~

.........................................................................................


check jupyter notebook/jupyterlab working or not.

####
