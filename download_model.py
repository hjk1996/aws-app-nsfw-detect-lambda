from transformers import pipeline


if __name__ == "__main__":
    classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection"
    )
    classifier.save_pretrained("./model")
