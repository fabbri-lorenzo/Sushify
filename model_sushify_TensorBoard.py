from classes_sushify import *

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup path to data folder
data_path = Path("Going_Modular/data/")
image_path = data_path / "sushi_types"
# image_path = data_path / "pizza_steak_sushi"


# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

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


# SETUP THE PRETRAINED MODEL with pretrained weights and send it to the target device (torchvision v0.13+)
model_sushify = efficientnet_b0(weights=weights)


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

# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


############################# Train and then visualisation with TENSORBOARD
# Create a writer with all default settings
writer = SummaryWriter()


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.writer.SummaryWriter,  # new parameter to take in a writer
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]}
      For example if training for epochs=2:
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = engine.train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = engine.test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            # Close the writer
            writer.close()
        else:
            pass
    ### End new ###

    # Return the filled results at the end of the epochs
    return results


# Train model
# Note: Not using engine.train() since the original script isn't updated to use writer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_sushify.parameters(), lr=0.001)

set_seeds()

NUM_EPOCHS = 3

# Start the timer
start_time = timer()

if __name__ == "__main__":
    results = train(
        model=model_sushify,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        writer=writer,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(
        f"[INFO] Total training time: {end_time-start_time:.3f} seconds , {((end_time-start_time)/60):.1f}minutes"
    )
