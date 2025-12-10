from huggingface_hub import list_repo_files
try:
    files = list_repo_files("underdogquality/yolo11s-pest-detection")
    print("Files in repo:")
    for f in files:
        print(f)
except Exception as e:
    print(f"Error: {e}")
