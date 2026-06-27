from __future__ import annotations

MODEL_NAME = "densenet121"
TARGET_LABELS = ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass"]
METADATA_COLUMNS = [
    "Image Index",
    "Patient ID",
    "split",
    "image_path",
    "Patient Gender",
    "View Position",
    "AgeBin",
    "has_out_of_scope_label",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MIN_TUNED_THRESHOLD_POSITIVES = 50
THRESHOLD_PRIOR_SEARCH_RADIUS = 0.10
LABEL_THRESHOLD_PRIORS = {
    "No Finding": 0.31707045435905457,
    "Infiltration": 0.46902427077293396,
    "Effusion": 0.5468099117279053,
    "Atelectasis": 0.36542871594429016,
    "Nodule": 0.6589330434799194,
    "Mass": 0.5728916525840759,
}
DYNAMIC_THRESHOLD_PRIOR_LABELS = {"Infiltration"}
