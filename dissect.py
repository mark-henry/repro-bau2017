import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.pool(self.gelu(self.conv1(x)))  # -> 14x14
        x = self.pool(self.gelu(self.conv2(x)))  # -> 7x7
        x = self.pool(self.gelu(self.conv3(x)))  # -> 3x3
        x = x.view(-1, 64 * 3 * 3)
        return self.fc(x)

def generate_primitives(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic primitive images in 2x2 grids for network dissection
    
    Returns:
        images: Tensor of shape (n_samples, 1, 28, 28) containing compound images
        labels: Tensor of shape (n_samples, 8) containing binary labels for each primitive type
        masks: Tensor of shape (n_samples, 8, 28, 28) containing pixel-wise masks for each primitive
    """
    images = np.zeros((n_samples, 28, 28))
    masks = np.zeros((n_samples, 8, 28, 28))  # 8 primitive types
    labels = np.zeros((n_samples, 8))  # Binary labels for each primitive type
    IMAGE_SIZE = 28
    RADIUS = 7  # Smaller radius for quarter-size primitives
    
    def draw_primitive(img: np.ndarray, mask: np.ndarray, shape_type: int, quadrant: int):
        """Draw a primitive in the specified quadrant (0=TL, 1=TR, 2=BL, 3=BR)"""
        # Calculate quadrant bounds
        x_start = (quadrant % 2) * 14
        y_start = (quadrant // 2) * 14
        
        if shape_type == 0:  # Diagonal right (\)
            for j in range(10):
                y = y_start + j + 2
                x = x_start + j + 2
                if 0 <= y < y_start + 14 and 0 <= x < x_start + 14:
                    img[y:y+2, x:x+2] = 1
                    mask[y:y+2, x:x+2] = 1
        
        elif shape_type == 1:  # Diagonal left (/)
            for j in range(10):
                y = y_start + j + 2
                x = x_start + 12 - j
                if 0 <= y < y_start + 14 and 0 <= x < x_start + 14:
                    img[y:y+2, x:x+2] = 1
                    mask[y:y+2, x:x+2] = 1
        
        elif shape_type == 2:  # Horizontal (-)
            y = y_start + 7
            img[y:y+2, x_start+2:x_start+12] = 1
            mask[y:y+2, x_start+2:x_start+12] = 1
            
        elif shape_type == 3:  # Vertical (|)
            x = x_start + 7
            img[y_start+2:y_start+12, x:x+2] = 1
            mask[y_start+2:y_start+12, x:x+2] = 1
            
        else:  # Curves
            # Define centers for each curve type
            if shape_type == 4:  # NE curve (╰)
                center = (x_start + 14, y_start)
            elif shape_type == 5:  # SE curve (╭)
                center = (x_start + 14, y_start + 14)
            elif shape_type == 6:  # SW curve (╮)
                center = (x_start, y_start + 14)
            else:  # NW curve (╯)
                center = (x_start, y_start)

            # Generate points along the full circle
            t = np.linspace(0, 2*np.pi, 40)
            for angle in t:
                x = int(center[0] + RADIUS * np.cos(angle))
                y = int(center[1] + RADIUS * np.sin(angle))
                if x_start <= x < x_start + 14 and y_start <= y < y_start + 14:
                    img[y, x] = 1
                    mask[y, x] = 1
                    # Make curve 2 pixels thick
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if x_start <= nx < x_start + 14 and y_start <= ny < y_start + 14:
                            img[ny, nx] = 1
                            mask[ny, nx] = 1
    
    for i in range(n_samples):
        # For each image, randomly select 4 primitives for the quadrants
        selected_primitives = np.random.randint(0, 8, size=4)
        for quad, prim in enumerate(selected_primitives):
            draw_primitive(images[i], masks[i, prim], prim, quad)
            labels[i, prim] = 1  # Mark this primitive as present
    
    return (torch.FloatTensor(images).unsqueeze(1), 
            torch.FloatTensor(labels),
            torch.FloatTensor(masks))

def train_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Train on MNIST with augmentation"""
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return model

def compute_binary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    Args:
        pred: Binary prediction tensor
        target: Binary ground truth tensor
    Returns:
        IoU score (1.0 if both pred and target are empty, as this indicates perfect agreement)
    """
    intersection = torch.logical_and(pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()
    # If both pred and target are empty (union=0), return 1.0 as they perfectly match
    return 1.0 if union == 0 else (intersection / union).item()

def get_activations(model: nn.Module, images: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Get activations for a specific layer with binary segmentation scoring"""
    activations = {}
    
    def hook(name):
        def hook_fn(module, input, output):
            # Store raw activations
            activations[name] = output.detach()
        return hook_fn
    
    # Register hooks
    if layer_name == 'conv1':
        model.conv1.register_forward_hook(hook('conv1'))
    elif layer_name == 'conv2':
        model.conv2.register_forward_hook(hook('conv2'))
    elif layer_name == 'conv3':
        model.conv3.register_forward_hook(hook('conv3'))
    
    with torch.no_grad():
        model(images)
    
    # Get raw activations
    raw_activations = activations[layer_name]
    
    # Compute threshold for top 0.5% of activations per unit (as per Bau et al.)
    batch_size = raw_activations.size(0)
    n_channels = raw_activations.size(1)
    spatial_size = raw_activations.size(2) * raw_activations.size(3)
    
    # Reshape for easier thresholding
    reshaped_acts = raw_activations.view(batch_size, n_channels, -1)
    thresholds = torch.quantile(reshaped_acts, 0.995, dim=2).unsqueeze(2).unsqueeze(3)
    
    # Create binary activation maps
    binary_activations = (raw_activations >= thresholds).float()
    
    # Upsample to match input resolution if needed
    if binary_activations.size(2) != 28:
        binary_activations = nn.functional.interpolate(
            binary_activations,
            size=(28, 28),
            mode='nearest'
        )
    
    return binary_activations

def visualize_detector_performance(model: nn.Module, layer_name: str, 
                                channel: int, primitive_idx: int,
                                images: torch.Tensor, masks: torch.Tensor,
                                device: torch.device):
    """Visualize detector performance with color overlays
    Reddish-Blue: Detector activation (false positives)
    Amber: Ground truth
    Green: Intersection (True Positive)
    
    Shows six examples, prioritizing:
    1. Perfect true negatives (empty vs empty)
    2. Small false positives
    3. True positives with varying amounts of overlap
    """
    # Get activations for this channel
    binary_acts = get_activations(model, images, layer_name)[:, channel]
    
    # Get ground truth for this primitive
    gt_mask = masks[:, primitive_idx]
    
    # Calculate overall IoU score for this detector
    overall_iou = compute_binary_iou(binary_acts, gt_mask)
    
    # Calculate coverage and intersection for each example
    gt_coverage = gt_mask.sum(dim=(1,2))
    act_coverage = binary_acts.sum(dim=(1,2))
    intersection = torch.logical_and(binary_acts, gt_mask).sum(dim=(1,2)).float()
    
    # Find different types of examples
    empty_mask = (gt_coverage == 0)
    empty_act = (act_coverage == 0)
    perfect_neg = torch.logical_and(empty_mask, empty_act)
    
    small_fp = torch.logical_and(empty_mask, torch.logical_and(act_coverage > 0, act_coverage < 10))
    has_tp = torch.logical_and(gt_coverage > 0, act_coverage > 0)
    
    # Get indices for each type, with fallbacks
    example_indices = []
    
    # Try to get one perfect negative
    perfect_neg_idx = torch.where(perfect_neg)[0]
    if len(perfect_neg_idx) > 0:
        example_indices.append((perfect_neg_idx[0], "Perfect Negative"))
    
    # Try to get small false positives
    small_fp_idx = torch.where(small_fp)[0]
    for i in range(min(2, len(small_fp_idx))):
        example_indices.append((small_fp_idx[i], "Small False Positive"))
    
    # Fill remaining slots with true positives or other examples
    tp_idx = torch.where(has_tp)[0]
    
    # Sort true positives by IoU score
    if len(tp_idx) > 0:
        tp_ious = intersection[tp_idx] / torch.maximum(gt_coverage[tp_idx] + act_coverage[tp_idx] - intersection[tp_idx], torch.ones_like(intersection[tp_idx]))
        sorted_tp = tp_idx[torch.argsort(tp_ious, descending=True)]
        for idx in sorted_tp[:6-len(example_indices)]:
            example_indices.append((idx, "True Positive"))
    
    # If we still need more examples, add any remaining examples
    if len(example_indices) < 6:
        remaining_idx = torch.arange(len(images))
        used_idx = torch.tensor([idx for idx, _ in example_indices])
        available_idx = remaining_idx[~torch.isin(remaining_idx, used_idx)]
        for idx in available_idx[:6-len(example_indices)]:
            example_indices.append((idx, "Other Example"))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    def plot_example(ax, idx, example_type):
        rgb = np.zeros((28, 28, 3))
        
        # Convert tensors to numpy for easier manipulation
        act = binary_acts[idx].cpu().numpy()
        gt = gt_mask[idx].cpu().numpy()
        
        # Reddish-Blue for detector activation (false positives)
        rgb[act > 0] = [0.3, 0, 0.9]  # More reddish blue
        
        # Amber for ground truth
        rgb[gt > 0] = [1.0, 0.75, 0]  # More amber than yellow
        
        # Green for the intersection (true positives)
        intersection = np.logical_and(act, gt)
        rgb[intersection] = [0, 1, 0]
        
        # Plot
        ax.imshow(images[idx].squeeze().cpu(), cmap='gray')
        ax.imshow(rgb, alpha=0.5)
        ax.axis('off')
        
        # Calculate IoU and coverage for this image
        iou = compute_binary_iou(binary_acts[idx], gt_mask[idx])
        gt_cov = gt_mask[idx].sum().item()
        act_cov = binary_acts[idx].sum().item()
        ax.set_title(f'{example_type}\nIoU: {iou:.3f}\nGT: {gt_cov:.0f} Act: {act_cov:.0f}')
    
    # Plot all examples
    for i, (idx, example_type) in enumerate(example_indices):
        plot_example(axes[i], idx, example_type)
    
    primitive_names = ['Diagonal Right', 'Diagonal Left', 'Horizontal', 'Vertical', 
                      'NE', 'SE', 'SW', 'NW']
    plt.suptitle(f'{layer_name} Channel {channel} - {primitive_names[primitive_idx]}\nOverall IoU: {overall_iou:.3f}',
                y=1.02)
    plt.tight_layout()
    plt.savefig(f'{layer_name}_ch{channel}_prim{primitive_idx}_performance.png',
                bbox_inches='tight')
    plt.close()

def analyze_neurons(model: nn.Module, device: torch.device):
    """Perform network dissection analysis using binary segmentation scoring"""
    # Generate primitives
    print("\nGenerating primitives for analysis...")
    images, labels, masks = generate_primitives(2000)
    images = images.to(device)
    masks = masks.to(device)
    
    # Get activations for each layer
    layers = ['conv1', 'conv2', 'conv3']
    primitive_names = ['Diagonal Right', 'Diagonal Left', 'Horizontal', 'Vertical', 
                      'NE', 'SE', 'SW', 'NW']
    
    for layer_name in layers:
        print(f"\nAnalyzing {layer_name}:")
        # Get binary activation maps
        binary_activations = get_activations(model, images, layer_name)
        
        # For each neuron, compute IoU with each primitive type
        n_channels = binary_activations.size(1)
        
        # Store detectors for visualization
        detectors = []  # (channel_idx, primitive_idx, iou_score)
        
        for channel in range(n_channels):
            channel_acts = binary_activations[:, channel]
            
            # For each primitive type
            for prim_idx in range(8):
                # Get ground truth masks for this primitive
                gt_masks = masks[:, prim_idx]
                
                # Compute IoU score
                iou_score = compute_binary_iou(channel_acts, gt_masks)
                
                # Report if IoU exceeds threshold
                if iou_score > 0.1:
                    print(f"Channel {channel} detects {primitive_names[prim_idx]} (IoU: {iou_score:.3f})")
                    detectors.append((channel, prim_idx, iou_score))
                    
                    # Visualize detector performance using all examples
                    visualize_detector_performance(model, layer_name, channel, prim_idx,
                                                images, masks, device)

def visualize_neuron_detectors(model: nn.Module, layer_name: str, 
                             top_detectors: List[Tuple[int, int, float]], 
                             images: torch.Tensor, labels: torch.Tensor):
    """Visualize the top detecting neurons and their maximally activating images"""
    # Sort by IoU score
    top_detectors = sorted(top_detectors, key=lambda x: x[2], reverse=True)[:5]  # Top 5
    
    fig, axes = plt.subplots(len(top_detectors), 4, figsize=(12, 3*len(top_detectors)))
    if len(top_detectors) == 1:
        axes = axes.reshape(1, -1)
    
    primitive_names = ['Diagonal Right', 'Diagonal Left', 'Horizontal', 'Vertical', 
                      'NE', 'SE', 'SW', 'NW']
    
    for i, (channel, prim_idx, iou) in enumerate(top_detectors):
        # Get images of this primitive type
        prim_mask = (labels == prim_idx)
        prim_images = images[prim_mask].squeeze()[:4]  # Get up to 4 examples
        
        # Plot the examples
        for j, img in enumerate(prim_images):
            axes[i, j].imshow(img.cpu().numpy(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(f"Ch {channel}\n{primitive_names[prim_idx]}\nIoU: {iou:.3f}")
    
    plt.tight_layout()
    plt.savefig(f'{layer_name}_detectors.png')
    plt.close()

def visualize_primitives(n_examples: int = 3):
    """Visualize examples of compound primitive images"""
    images, labels, masks = generate_primitives(n_examples)
    
    # Create a grid of examples
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 4, 8))
    
    # Show original images
    for j in range(n_examples):
        axes[0, j].imshow(images[j].squeeze(), cmap='gray')
        axes[0, j].axis('off')
        axes[0, j].set_title('Compound Image')
    
    # Show masks (overlay all primitives with different colors)
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'orange']
    for j in range(n_examples):
        # Start with black background
        rgb = np.zeros((28, 28, 3))
        # Add each primitive mask with its color
        for prim_idx in range(8):
            if labels[j, prim_idx]:  # If this primitive is present
                mask = masks[j, prim_idx].numpy()
                color = np.array(plt.cm.colors.to_rgb(colors[prim_idx]))
                rgb[mask > 0] = color
        
        axes[1, j].imshow(rgb)
        axes[1, j].axis('off')
        axes[1, j].set_title('Primitive Masks')
    
    plt.tight_layout()
    plt.savefig('primitives.png')
    plt.close()

if __name__ == "__main__":
    print("Generating primitive examples...")
    visualize_primitives(4)    
    
    # Then proceed with the analysis
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    # Try to load existing model, train if not found
    if os.path.exists('model.pth'):
        print("\nLoading existing model...")
        model.load_state_dict(torch.load('model.pth'))
    else:
        print("\nTraining new model...")
        model = train_model(model, device)
        print("Saving model...")
        torch.save(model.state_dict(), 'model.pth')
    
    print("\nAnalyzing neurons...")
    analyze_neurons(model, device)
