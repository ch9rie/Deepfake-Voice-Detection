import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio # --- ADDED for augmentation
from app import AudioCNN

def train_model():
    """Function to train the audio classification model with anti-overfitting techniques."""
    print("Loading datasets...")
    try:
        train_data = torch.load('data/for_2_sec_train.pt')
        valid_data = torch.load('data/for_2_sec_valid.pt')
    except FileNotFoundError:
        print("\n" + "="*50)
        print("FATAL ERROR: Dataset files not found in the 'data' folder!")
        print("Please run 'python create_dataset.py' first.")
        print("="*50 + "\n")
        return
    
    X_train, y_train = train_data['data'], train_data['labels']
    X_valid, y_valid = valid_data['data'], valid_data['labels']
    
    train_dataset = TensorDataset(X_train, y_train.float().unsqueeze(1))
    valid_dataset = TensorDataset(X_valid, y_valid.float().unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=0)

    print("Initializing model...")
    model = AudioCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.BCELoss()
    # --- MODIFIED: Added weight_decay for L2 regularization ---
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # --- ADDED: Learning rate scheduler to reduce LR over time ---
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- ADDED: Data augmentation transforms ---
    # These will be applied randomly to each batch during training.
    freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)

    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs on device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # --- ADDED: Apply data augmentation on the fly ---
            # This creates new variations of the data in each epoch.
            augmented_inputs = time_masking(freq_masking(inputs))

            optimizer.zero_grad()
            outputs = model(augmented_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- MODIFIED: Validation loop now also calculates and prints loss ---
        model.eval()
        valid_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate validation loss
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # --- MODIFIED: More detailed print statement ---
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {accuracy:.2f}%")

        # --- ADDED: Step the scheduler ---
        scheduler.step()

    torch.save(model.state_dict(), "model.pth")
    print("\nTraining complete. Model saved to model.pth")

if __name__ == "__main__":
    train_model()