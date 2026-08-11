import numpy as np
import matplotlib
import torch
import os
import torch.nn.functional as F
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textwrap import wrap
import open3d as o3d


def compute_similarity(top_k_mask_text_clip_feat, reshot_text_clip_feat):
    top_k_mask_text_clip_feat = [
        torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat
        for feat in top_k_mask_text_clip_feat
    ]
    
    if isinstance(reshot_text_clip_feat, np.ndarray):
        reshot_text_clip_feat = torch.from_numpy(reshot_text_clip_feat)
        
    mask_text_clip_tensor = torch.stack(top_k_mask_text_clip_feat)  # (k, 1, D)
    reshot_text_clip_tensor = reshot_text_clip_feat.unsqueeze(0)   # (1, 1, D)
    
    mask_text_clip_tensor = mask_text_clip_tensor.squeeze(1)  # (k, D)
    reshot_text_clip_tensor = reshot_text_clip_tensor.squeeze(1)  # (1, D)

        
    similarities = F.cosine_similarity(mask_text_clip_tensor, reshot_text_clip_tensor.expand_as(mask_text_clip_tensor), dim=1) # (k,)
    
    return similarities

# def compute_similarity(top_k_mask_text_clip_feat, reshot_text_clip_feat):
#     top_k_mask_text_clip_feat = [
#         torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat
#         for feat in top_k_mask_text_clip_feat
#     ]
    
#     if isinstance(reshot_text_clip_feat, np.ndarray):
#         reshot_text_clip_feat = torch.from_numpy(reshot_text_clip_feat)
        
#     mask_text_clip_tensor = torch.stack(top_k_mask_text_clip_feat)  # (k, 1, D)
#     reshot_text_clip_tensor = reshot_text_clip_feat.unsqueeze(-1)   # (1, D, 1)
    
#     mask_text_clip_tensor = mask_text_clip_tensor.squeeze(1).unsqueeze(-1)  # (k, D, 1)

        
#     similarities = F.cosine_similarity(mask_text_clip_tensor, reshot_text_clip_tensor, dim=1) # (k,)
    
#     return similarities

def plot_images_with_background(images, masks, background, savedir, idx_obj):
    os.makedirs(savedir, exist_ok=True)
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Background: {background}"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, similarity, pointcloud, savedir, idx_obj):
    os.makedirs(savedir, exist_ok=True)
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}\nSimilarity: {similarity[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()
    o3d.io.write_point_cloud( savedir / f"{idx_obj}.ply", pointcloud)


def get_bbox_center(bbox):
    # bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
    if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        return np.array(bbox.get_center())
    elif isinstance(bbox, (list, np.ndarray)):
        return np.array([(bbox[0]+bbox[3])/2,
                     (bbox[1]+bbox[4])/2,
                     (bbox[2]+bbox[5])/2])
    else:
        raise TypeError(f"Unsupported bbox type: {type(bbox)}")

from PIL import Image

def concat_images(img1, img2, direction="horizontal"):
    """
    :param img1: PIL.Image
    :param img2: PIL.Image
    :param direction: "horizontal" or "vertical"
    """
    if direction == "horizontal":
        # The height is the maximum, and the insufficient height is filled with a black background
        new_height = max(img1.height, img2.height)
        new_width = img1.width + img2.width
        new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
    else:  # vertical
        new_width = max(img1.width, img2.width)
        new_height = img1.height + img2.height
        new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.height))
    
    return new_img

from PIL import Image

def concat_images_resize(img1, img2, direction="horizontal"):
    """
    resize
    :param img1: PIL.Image
    :param img2: PIL.Image
    :param direction: "horizontal" or "vertical"
    """
    if direction == "horizontal":
        # same height
        target_height = min(img1.height, img2.height)
        img1_resized = img1.resize((int(img1.width * target_height / img1.height), target_height))
        img2_resized = img2.resize((int(img2.width * target_height / img2.height), target_height))

        new_width = img1_resized.width + img2_resized.width
        new_img = Image.new("RGB", (new_width, target_height))
        new_img.paste(img1_resized, (0, 0))
        new_img.paste(img2_resized, (img1_resized.width, 0))

    else:  # vertical
        # same width
        target_width = min(img1.width, img2.width)
        img1_resized = img1.resize((target_width, int(img1.height * target_width / img1.width)))
        img2_resized = img2.resize((target_width, int(img2.height * target_width / img2.width)))

        new_height = img1_resized.height + img2_resized.height
        new_img = Image.new("RGB", (target_width, new_height))
        new_img.paste(img1_resized, (0, 0))
        new_img.paste(img2_resized, (0, img1_resized.height))

    return new_img

def plot_final_caption(index, images, caption, savedir, prefix=""):
    """
    Visualize two images side by side with a shared caption.
    
    Args:
        images: list of two images (PIL.Image or numpy array)
        caption: string, shared caption for both images
        savedir: directory to save the figure
        filename: saved filename
    """
    os.makedirs(savedir, exist_ok=True)
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(2):
        ax = axarr[i]
        ax.imshow(images[i])
        ax.axis('off')

    # Wrap caption and put as a suptitle
    wrapped_caption = '\n'.join(wrap(caption, 80))
    fig.suptitle(wrapped_caption, fontsize=12, y=0.05) 

    plt.tight_layout(rect=[0, 0.1, 1, 1])  
    plt.savefig(savedir / f"{prefix}_{index}.png")
    plt.close()
