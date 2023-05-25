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

~~~~~~~

### Package for Data Prepocessing
from city_sustainability.preprocessing import image_resize

### Package for Data Loading
from city_sustainability.loading_data import loading_paths

### Package for Data Loading in batches
from city_sustainability.loading_batch import image_and_label_arrays_batch

# Import all image array as X and label array as y

image_paths = [...]  # List of image file paths
label_paths = [...]  # List of label file paths

X, y = image_and_label_arrays_batch(image_paths, label_paths, sampling_ratio=1, batch_size=32)

~~~~~~~

### Package for Model
~~~~~

from city_sustainability.models import unet_model

### Create the model:-)
model = unet_model.build_model(input_shape=(a, b, 3), num_classes=c)

### Compile the model:-)
unet_model.compile_model(model)

### Train the model:-)
history = unet_model.train_model(model, x_train , y_train  , epochs=1, batch_size=512, validation_split=0.2)

### Evaluate the model:-)
unet_model.evaluate_model(model, x_test, y_test)

### Make predictions:-)
predictions = unet_model.predict(model, x_test)

~~~~~

.........................................................................................
### Streamlit
.........................................................................................

install streamlit by running requirements

streamlit run Home.py

.........................................................................................

check jupyter notebook/jupyterlab working or not.

####
