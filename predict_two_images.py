from ultralytics import YOLO
import torch


def main():
    # Load your trained model
    model = YOLO(r"runs\segment\runs\addis_yolov8m_seg\weights\best.pt")

    # Check GPU
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Using CPU")
        device = "cpu"

    # Predict on the 2 images
    image_paths = [
        r"test_images\image1.png",
        r"test_images\image2.png"
    ]

    results = model.predict(
        source=image_paths,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        device=device,
        project="runs/predict",
        name="two_test_images",
        exist_ok=True
    )

    # Print prediction summary
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {image_paths[i]}")
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]
                print(f"  {class_name}: {conf:.3f}")
        else:
            print("  No objects detected.")


if __name__ == "__main__":
    main()