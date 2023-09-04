#%%capture
#!pip install super-gradients
#!pip install imutils
#!pip install roboflow
#!pip install pytube --upgrade

#!pip install torchinfo

from super_gradients.training import models
from torchinfo import summary

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

summary(model=yolo_nas_l,
        input_size=(16, 3, 640, 640),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

url = "https://previews.123rf.com/images/freeograph/freeograph2011/freeograph201100150/158301822-group-of-friends-gathering-around-table-at-home.jpg"
predictions = yolo_nas_l.predict(url, conf=0.25)

# Get the predicted objects
prediction_objects = list(predictions._images_prediction_lst)[0]

# Get the bounding boxes
bboxes = prediction_objects.prediction.bboxes_xyxy

# Get the class labels
int_labels = prediction_objects.prediction.labels.astype(int)
class_names = prediction_objects.class_names
pred_classes = [class_names[i] for i in int_labels]

# # Print the predicted objects
# for i, bbox in enumerate(bboxes):
#     print(f"Object {i + 1}: {pred_classes[i]}, {bbox}")

# Get the class labels
int_labels = prediction_objects.prediction.labels.astype(int)
class_names = prediction_objects.class_names
pred_classes = [class_names[i] for i in int_labels]

# Create a dictionary to store the number of objects for each category
object_counts = {}
for pred_class in pred_classes:
    if pred_class not in object_counts:
        object_counts[pred_class] = 0
    object_counts[pred_class] += 1

# Print the number of objects for each category
for pred_class, count in object_counts.items():
    # print(f"The number of {pred_class} objects in the image is: {count}")
    print(f"There are {count} {pred_class}")


