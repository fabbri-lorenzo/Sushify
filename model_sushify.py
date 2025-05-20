from classes_sushify import *

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup path to data folder
data_path = Path("Modular/data/")
image_path = data_path / "sushi_types"
# image_path = data_path / "pizza_steak_sushi"


# Listing the content of data directory in order to get familiar with data
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# walk_through_dir(image_path)


# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"


# Visualize an image of dataset with some details

# Set seed
random.seed(42)  

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# img.show()

####################
# Get a set of pretrained model weights
weights = (
    EfficientNet_B0_Weights.DEFAULT
)  # .DEFAULT = best available weights from pretraining on ImageNet

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,  # perform same data transforms on our own data as the pretrained model
    batch_size=32,
)  # set mini-batch size to 32

# print(train_dataloader, test_dataloader, class_names)

# SETUP THE PRETRAINED MODEL with pretrained weights and send it to the target device (torchvision v0.13+)
model_sushify = efficientnet_b0(weights=weights)


# Print a summary using torchinfo (uncomment for actual output)

# summary(
#   model=model_sushify,
#   input_size=(
#       32,
#       3,
#       224,
#       224,
#   ),  # make sure this is "input_size", not "input_shape"
#   # col_names=["input_size"], # uncomment for smaller output
#   col_names=["input_size", "output_size", "num_params", "trainable"],
#   col_width=20,
#   row_settings=["var_names"],)


# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model_sushify.features.parameters():
    param.requires_grad = False

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model_sushify.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(
        in_features=1280,
        out_features=output_shape,  # same number of output units as our number of classes
        bias=True,
    ),
).to(device)

# Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
#
# summary(
#        model_sushify,
#        input_size=(
#            32,
#            3,
#            224,
#            224,
#        ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
#        verbose=0,
#        col_names=["input_size", "output_size", "num_params", "trainable"],
#        col_width=20,
#        row_settings=["var_names"],
#    )


# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == "__main__":
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_sushify.parameters(), lr=0.001)
    # Start the timer
    start_time = timer()

    # Setup training and save the results
    results = engine.train(
        model=model_sushify,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=2,
        device=device,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(
        f"[INFO] Total training time: {end_time-start_time:.3f} seconds , {((end_time-start_time)/60):.1f}minutes"
    )

    # Plot the loss curves of our model
    plot_loss_curves(results)
    plt.show()
