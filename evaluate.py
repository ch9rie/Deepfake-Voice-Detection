import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app import AudioCNN # Import the same model architecture

def evaluate_model():
    """Loads the trained model and evaluates its performance on the test set."""
    
    print("Loading the trained model and test dataset...")
    try:
        model = AudioCNN()
        model.load_state_dict(torch.load("model.pth"))
        test_data = torch.load("data/for_2_sec_test.pt")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please make sure 'model.pth' and 'data/for_2_sec_test.pt' exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test, y_test = test_data['data'], test_data['labels']
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    all_preds = []
    all_labels = []

    print("Running inference on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            predicted_probs = outputs.cpu().numpy()
            predicted_labels = (predicted_probs > 0.5).astype(int)
            
            all_preds.extend(predicted_labels.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # --- Calculate Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n" + "="*30)
    print("      Model Evaluation Report")
    print("="*30)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print("="*30 + "\n")

    # --- Create and Display Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    evaluate_model()
