import torch
import pytorchvideo.models.resnet as resnet

def create_resnet():
    # Create a 3D ResNet-50 model
    model = resnet.create_resnet(
        input_channel=3,
        model_depth=50,
        model_num_class=400,  # Kinetics classes, will be removed
        norm=torch.nn.BatchNorm3d,
        activation=torch.nn.ReLU,
    )

    # Remove the final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Save the modified model
    torch.save(model.state_dict(), "../models/3d_resnet/resnet3d_50_features.pt")
    
if __name__ == "__main__":
    create_resnet()