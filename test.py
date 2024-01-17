from glob import glob

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

if __name__ == "__main__":
    safe_images = glob("./samples/safe/*")
    not_safe_images = glob("./samples/not_safe/*")
    total_images = len(safe_images) + len(not_safe_images)
    passed = 0
    failed = 0
    model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

    for img in safe_images:
        img = Image.open(img)
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
        print(logits)
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        print(label)
        if label == "normal":
            passed += 1
    for img in not_safe_images:
        img = Image.open(img)
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
        print(logits)
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        print(label)
        if label == "nsfw":
            passed += 1

    print(f"Passed: {passed}/{total_images}")
