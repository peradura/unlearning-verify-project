from upload.unlearned_dataset.manager import UnlearningDatasetManager

def main():
    
    manager = UnlearningDatasetManager()
    result = manager.register(
        dataset="example",
        name="Example Dataset",
        version="1.0"
    )

if __name__ == "__main__":
    main()
    