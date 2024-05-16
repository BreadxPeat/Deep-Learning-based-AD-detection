from google.colab import drive
drive.mount('/content/drive')

custom_image_path = r'__mention path to your testing set__'
custom_image_path1 = r'__mention path to your testing set___'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x/255.0)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root=custom_image_path, transform=transform)
dataset_test = ImageFolder(root=custom_image_path1, transform=transform)

data_iter = iter(train_loader)
images, labels = next(data_iter)

# Accessing the dimensions of the batch of images
batch_size, num_channels, height, width = images.size()
print("Batch size:", batch_size)
print("Image dimensions:", (num_channels, height, width))

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)