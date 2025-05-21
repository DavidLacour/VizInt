import torch
import numpy as np

def compute_mahalanobis_stats(model, train_dataloader, device, feature_extractor_layer):
    """
    Computes class centroids and the inverse covariance matrix
    from the training data features.

    Args:
        model: The trained Vision Transformer model.
               Assumes it has an evaluation mode and can extract features
               from a specified layer before the final classification head.
        train_dataloader: DataLoader for the training dataset.
                          Should yield (images, labels).
        device: The device to run the model on (e.g., 'cuda', 'cpu').
        feature_extractor_layer: The name or reference to the layer
                                 from which to extract features (embeddings).
                                 This depends on your model's architecture.
                                 For a ViT, this is usually the layer before
                                 the final classification linear layer, e.g.,
                                 the output associated with the [CLS] token
                                 or a global pooling layer.

    Returns:
        centroids (dict): A dictionary mapping class labels (int) to their
                          pre-computed centroid (numpy array).
        inv_cov_matrix (numpy.ndarray): The inverse covariance matrix.
    """
    model.eval() # Set model to evaluation mode
    all_features = []
    all_labels = []

    # --- Step 1: Extract features for all training images ---
    # You'll need to modify this part based on how to access
    # features from your specific ViT model implementation.
    # A common way is to register hooks or call a forward pass
    # function that returns intermediate features.

    # Example placeholder for feature extraction logic:
    # Let's assume a hypothetical function in your model/wrapper
    # called `extract_features(image_tensor)` which returns the
    # feature vector from `feature_extractor_layer`.
    # If you are using Hugging Face Transformers, you might need
    # to configure the model's forward pass or use hooks.

    feature_dim = None # To be determined from extracted features

    print("Extracting features from training data...")
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in train_dataloader:
            images = images.to(device)
            # labels = labels.to(device) # Labels are needed on CPU later for stats

            # --- Replace this with your actual feature extraction ---
            # Example using a hypothetical method:
            # features = model.extract_features(images)
            # Or using a hook:
            # features = [] # Store features from hook
            # hook = feature_extractor_layer.register_forward_hook(
            #     lambda module, input, output: features.append(output.cpu().numpy())
            # )
            # model(images) # Forward pass to trigger hook
            # hook.remove()
            # features = features[0] # Assuming the hook output is a list of one tensor

            # A simple example if the model forward returns features directly
            # before the head (might require modifying the model's forward method)
            # features = model(images, return_features=True) # Hypothetical call
            # features = features.cpu().numpy()

            # *** Placeholder: Using a dummy feature extractor for demonstration ***
            # In a real scenario, replace this with code that gets the
            # actual features from your ViT before the classification head.
            # This dummy assumes input images are batches of flattened vectors
            # and the model just passes them through (not realistic for images)
            # Or, more realistically, assumes images have a fixed shape
            # and the model outputs a vector per image.
            batch_size = images.shape[0]
            if feature_dim is None:
                 # Guessing feature_dim. Needs to be correct for your model.
                 # e.g., for ViT-Base, it might be 768.
                 # Let's use a dummy random feature for now
                 feature_dim = 768 # Example dimension, replace with actual
                 print(f"Assuming feature dimension: {feature_dim}")
            features = np.random.rand(batch_size, feature_dim) # Replace with real features
            # *******************************************************************


            all_features.append(features)
            all_labels.append(labels.numpy()) # Move labels to CPU for numpy

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Extracted features for {all_features.shape[0]} training samples.")

    # --- Step 2: Calculate Class Centroids (μc) ---
    print("Calculating class centroids...")
    centroids = {}
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        class_features = all_features[all_labels == label]
        centroids[label] = np.mean(class_features, axis=0)
        print(f"  Calculated centroid for class {label} with {class_features.shape[0]} samples.")

    # --- Step 3: Calculate Centered Features ---
    print("Centering features by subtracting class centroids...")
    centered_features = np.copy(all_features)
    for label in unique_labels:
         centered_features[all_labels == label] -= centroids[label]


    # --- Step 4: Calculate Covariance Matrix (Σ) ---
    # np.cov expects variables as rows (shape M x N, M variables, N observations)
    # or observations as rows (shape N x M, N observations, M variables) with rowvar=False.
    # Our `centered_features` is N observations x M features, so use rowvar=False.
    print("Calculating covariance matrix...")
    cov_matrix = np.cov(centered_features, rowvar=False)
    print(f"Covariance matrix shape: {cov_matrix.shape}")


    # --- Step 5: Calculate Inverse Covariance Matrix (Σ⁻¹) ---
    print("Calculating inverse covariance matrix...")
    # Add a small epsilon for numerical stability, especially if feature_dim is large
    epsilon = 1e-6
    reg_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
    inv_cov_matrix = np.linalg.inv(reg_cov_matrix)
    # Alternatively, use pseudo-inverse:
    # inv_cov_matrix = np.linalg.pinv(cov_matrix)

    print("Mahalanobis stats computed.")
    return centroids, inv_cov_matrix

def mahalanobis_score(feature_vector, centroids, inv_cov_matrix):
    """
    Computes the Mahalanobis score for a single feature vector.

    Args:
        feature_vector (numpy.ndarray): The feature vector (embedding)
                                        of the input image. Shape (feature_dim,).
        centroids (dict): Dictionary of pre-computed class centroids.
        inv_cov_matrix (numpy.ndarray): The pre-computed inverse covariance matrix.

    Returns:
        float: The minimum squared Mahalanobis distance to any class centroid.
               This serves as the OOD score. Higher score means more likely OOD.
    """
    min_distance_sq = float('inf')

    for label, centroid in centroids.items():
        # Calculate difference vector: (feature_vector - μc)
        diff = feature_vector - centroid

        # Calculate Mahalanobis distance squared: diffᵀ Σ⁻¹ diff
        # np.dot handles dot product correctly for 1D arrays.
        # This computes (1, feature_dim) @ (feature_dim, feature_dim) @ (feature_dim, 1)
        distance_sq = np.dot(np.dot(diff, inv_cov_matrix), diff)

        # Update minimum distance
        min_distance_sq = min(min_distance_sq, distance_sq)

    return min_distance_sq

# --- Example Usage (Conceptual) ---

if __name__ == '__main__':
    # This part is conceptual and requires a trained model and dataloaders

    # 1. Assume you have a trained ViT model
    # model = YourTrainedViTModel(...)
    # model.to(device) # assuming device = 'cuda' or 'cpu'

    # 2. Assume you have training and testing dataloaders
    # train_dataloader = YourTrainDataLoader(...)
    # test_dataloader = YourTestDataLoader(...) # Should include ID and OOD images

    # 3. Define which layer's output to use as features
    # This requires inspecting your model's architecture
    # For a Hugging Face ViT, it might be something like:
    # feature_layer = model.vit.pooler # Or the output of the last transformer layer

    # *** Placeholder for actual model and dataloader setup ***
    print("--- Conceptual Example ---")
    print("Please replace this section with your actual model and data loading.")

    # Dummy data for demonstration:
    # Simulate extracting features for 100 training samples across 3 classes
    # and 10 test samples
    dummy_feature_dim = 768
    num_train_samples = 100
    num_classes = 3
    num_test_samples = 10

    # Simulate training data features and labels
    dummy_train_features = np.random.rand(num_train_samples, dummy_feature_dim)
    dummy_train_labels = np.random.randint(0, num_classes, num_train_samples)
    dummy_train_dataset = list(zip([None]*num_train_samples, dummy_train_labels)) # Images are None, just need labels
    # Need dummy features associated with labels for stats calculation
    dummy_train_features_dict = {lbl: [] for lbl in range(num_classes)}
    for i in range(num_train_samples):
        dummy_train_features_dict[dummy_train_labels[i]].append(dummy_train_features[i])

    # Simulate test data features (mixture of ID and OOD)
    # Let's make the first 5 ID-like (close to a centroid) and last 5 OOD-like (far)
    dummy_test_features = np.random.rand(num_test_samples, dummy_feature_dim)
    # Make first few features closer to centroid 0
    if num_test_samples >= 5 and num_train_samples > 0:
        dummy_test_features[:5] = dummy_train_features[0] + np.random.randn(5, dummy_feature_dim) * 0.1 # Close to centroid 0
    # Make last few features clearly different
    if num_test_samples > 5:
         dummy_test_features[5:] = dummy_test_features[5:] * 5 + 10 # Scale and shift away


    # *** End Placeholder ***

    # 4. Compute Mahalanobis statistics from training data
    # Need to adapt compute_mahalanobis_stats to use your actual model/dataloader
    # For the dummy example, we will manually create the required inputs
    print("\nCalculating stats using dummy data...")

    # Manual centroid calculation for dummy data
    dummy_centroids = {}
    for label in range(num_classes):
         class_features = np.array(dummy_train_features_dict[label])
         if len(class_features) > 0:
              dummy_centroids[label] = np.mean(class_features, axis=0)
         else:
              dummy_centroids[label] = np.zeros(dummy_feature_dim) # Handle empty class if any

    # Manual centered features calculation for dummy data
    dummy_centered_features = np.copy(dummy_train_features)
    for label in range(num_classes):
        if label in dummy_centroids:
             dummy_centered_features[dummy_train_labels == label] -= dummy_centroids[label]

    # Manual covariance and inverse calculation for dummy data
    dummy_cov_matrix = np.cov(dummy_centered_features, rowvar=False)
    epsilon = 1e-6
    reg_dummy_cov_matrix = dummy_cov_matrix + np.eye(dummy_cov_matrix.shape[0]) * epsilon
    dummy_inv_cov_matrix = np.linalg.inv(reg_dummy_cov_matrix)

    # In a real scenario, you'd call:
    # centroids, inv_cov_matrix = compute_mahalanobis_stats(model, train_dataloader, device, feature_layer)
    centroids = dummy_centroids # Use dummy stats for the example
    inv_cov_matrix = dummy_inv_cov_matrix


    # 5. Score new test images
    print("\nScoring dummy test images...")
    ood_scores = []
    for i, test_feature in enumerate(dummy_test_features):
        score = mahalanobis_score(test_feature, centroids, inv_cov_matrix)
        ood_scores.append(score)
        print(f"Test sample {i+1} Mahalanobis score: {score:.4f}")

    # 6. Interpret scores (conceptual)
    print("\nInterpreting scores (conceptual):")
    # You would typically set a threshold based on a validation set
    # For this dummy data, we expect lower scores for the first ~5 samples
    # and higher scores for the last ~5 samples.
    # Threshold setting is a critical step in practice!
    threshold = np.median(ood_scores) # Just for illustration
    print(f"Conceptual Threshold (median score): {threshold:.4f}")

    for i, score in enumerate(ood_scores):
        if score >= threshold: # Using >= for OOD as per the first paper's decision rule form
             print(f"Test sample {i+1} (Score: {score:.4f}) -> Detected as OOD")
        else:
             print(f"Test sample {i+1} (Score: {score:.4f}) -> Detected as In-Domain")

    # In a real application, you would use metrics like AUROC/AUPROOD
    # to evaluate the performance of these scores across a test set
    # with known ID/OOD labels.