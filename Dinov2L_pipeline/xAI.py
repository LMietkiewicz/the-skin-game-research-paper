# xAI.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from torch.nn import functional as F
from typing import Optional, List, Tuple, Union
import random
from tqdm import tqdm
import math

class DINOv2Attention:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.attention_output = None
        
        # Register hooks
        self.register_hooks()
        
    def register_hooks(self):
        """Register hook to capture attention output."""
        def attention_hook(module, input, output):
            self.attention_output = output
        
        # Register the hook
        self.hooks.append(self.target_layer.attention.output.register_forward_hook(attention_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_map(self, attention_output: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        """
        Process attention output into a spatial attention map.
        """
        # Get attention output and process it
        attention = attention_output.squeeze()  # Remove batch dimension
        
        # Calculate number of patches
        patch_size = 16  # DINOv2 default patch size
        n_patches_h = input_size[0] // patch_size
        n_patches_w = input_size[1] // patch_size
        
        # Take the mean over the feature dimension
        attention_weights = attention.mean(dim=-1)  # Average over features
        
        # Reshape into spatial grid, excluding the CLS token
        try:
            attention_map = attention_weights[1:].reshape(n_patches_h, n_patches_w)
        except RuntimeError:
            # Fallback to square grid if reshape fails
            n_tokens = attention_weights.shape[0] - 1  # Exclude CLS token
            grid_size = int(math.sqrt(n_tokens))
            attention_map = attention_weights[1:].reshape(grid_size, grid_size)
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate attention map for input image.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class (optional)
            
        Returns:
            Tuple of (attention_map, model_outputs)
        """
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Get attention map
        attention_map = self.get_attention_map(
            self.attention_output,
            input_size=(input_tensor.shape[2], input_tensor.shape[3])
        )
        
        # Resize to input image size
        attention_map = F.interpolate(
            attention_map,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return attention_map, outputs.logits

class XAIVisualizer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize attention visualization
        self.attention_vis = DINOv2Attention(
            model=self.model,
            target_layer=self.model.dinov2.encoder.layer[-1]
        )

    def generate_heatmap(
        self,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        alpha: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate heatmap overlay and raw attention map."""
        # Convert attention map to numpy
        attention_map = attention_map.squeeze().cpu().numpy()
        attention_map = cv2.resize(attention_map, (image.shape[2], image.shape[1]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Prepare original image
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create overlay
        overlay = (1 - alpha) * img_np + alpha * heatmap / 255
        overlay = np.clip(overlay, 0, 1)
        
        return overlay, attention_map

    def visualize_sample(
        self,
        image: torch.Tensor,
        true_class: int,
        pred_class: int,
        class_names: List[str],
    ) -> plt.Figure:
        """Generate visualization for a sample image."""
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax1.imshow(img_np)
        title = f'Original Image\nTrue: {class_names[true_class]}'
        if true_class != pred_class:
            title += f'\nPredicted (INCORRECT): {class_names[pred_class]}'
        else:
            title += f'\nPredicted (correct): {class_names[pred_class]}'
        ax1.set_title(title, fontsize=10, pad=10)
        ax1.axis('off')
        
        # Generate attention visualization
        attention_map, logits = self.attention_vis(image.unsqueeze(0).to(self.device))
        overlay, raw_attention = self.generate_heatmap(image, attention_map)
        
        # Get confidence scores
        probs = F.softmax(logits, dim=1)[0]
        true_conf = probs[true_class].item()
        pred_conf = probs[pred_class].item()
        
        # Plot attention overlay
        ax2.imshow(overlay)
        ax2.set_title(f'Attention Overlay\nTrue class conf: {true_conf:.2f}\nPred class conf: {pred_conf:.2f}',
                     fontsize=10, pad=10)
        ax2.axis('off')
        
        # Plot raw attention heatmap
        im = ax3.imshow(raw_attention, cmap='jet')
        ax3.set_title('Raw Attention Map', fontsize=10, pad=10)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        return fig

    def analyze_predictions(
        self,
        dataset,
        class_names: List[str],
        n_correct: int = 3,
        n_incorrect: int = 3,
        save_dir: Optional[str] = None
    ) -> None:
        """Analyze and visualize model predictions."""
        # Get predictions
        all_preds = []
        all_labels = []
        all_indices = []
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        print("Getting predictions...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                outputs = self.model(batch['pixel_values'].to(self.device))
                predictions = outputs.logits.argmax(dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_indices.extend(range(i * 32, min((i + 1) * 32, len(dataset))))
        
        # Separate correct and incorrect predictions
        correct_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l]
        incorrect_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
        
        print(f"\nFound {len(correct_indices)} correct and {len(incorrect_indices)} incorrect predictions")
        
        # Sample indices
        selected_correct = random.sample(correct_indices, min(n_correct, len(correct_indices)))
        selected_incorrect = random.sample(incorrect_indices, min(n_incorrect, len(incorrect_indices)))
        
        # Visualize samples
        def process_samples(indices, prefix):
            print(f"\nGenerating visualizations for {prefix} predictions...")
            for i, idx in enumerate(tqdm(indices)):
                image = dataset[all_indices[idx]]['pixel_values']
                true_class = all_labels[idx]
                pred_class = all_preds[idx]
                
                fig = self.visualize_sample(image, true_class, pred_class, class_names)
                if save_dir:
                    plt.savefig(f"{save_dir}/{prefix}_prediction_{i}.png", 
                              bbox_inches='tight', dpi=300)
                plt.close()
        
        process_samples(selected_correct, "correct")
        process_samples(selected_incorrect, "incorrect")