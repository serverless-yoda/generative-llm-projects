# setup_dataset.py
from roboflow import Roboflow
from config import ROBOFLOW_CONFIG

def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_CONFIG["api_key"])
    project = rf.workspace(ROBOFLOW_CONFIG["workspace"]).project(ROBOFLOW_CONFIG["project_name"])
    dataset = project.version(ROBOFLOW_CONFIG["version"]).download("yolov5")
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_dataset()