import os
import torch
import torchaudio
import random
from tqdm import tqdm

def preprocess_wav(file_path):
    """Loads and preprocesses a single WAV file into a mel spectrogram."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        target_len = 2 * 16000
        waveform = waveform[:, :target_len] if waveform.shape[1] > target_len else torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)(waveform)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")
        return None

def create_dataset_tensors(data_dir, output_file, max_files_per_class=5000):
    """Processes WAV files and saves them as a PyTorch tensor file."""
    print(f"Processing data in: {data_dir}")
    all_specs, all_labels = [], []
    
    real_path = os.path.join(data_dir, 'real')
    fake_path = os.path.join(data_dir, 'fake')
    
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.wav')]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith('.wav')]
    
    # To speed things up, we can limit the number of files we process
    files_to_process = random.sample(real_files, min(len(real_files), max_files_per_class)) + \
                       random.sample(fake_files, min(len(fake_files), max_files_per_class))
    
    print(f"Processing {len(files_to_process)} files...")

    for file_path in tqdm(files_to_process, desc=f"Creating {os.path.basename(output_file)}"):
        spec = preprocess_wav(file_path)
        if spec is not None:
            all_specs.append(spec)
            all_labels.append(0 if 'real' in file_path else 1)

    if not all_specs:
        print("No audio files were successfully processed.")
        return

    data_tensor = torch.stack(all_specs)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    torch.save({'data': data_tensor, 'labels': labels_tensor}, output_file)
    print(f"Successfully created dataset: {output_file}")

if __name__ == '__main__':
    # !!! IMPORTANT: UPDATE THIS PATH to where you unzipped the dataset !!!
    base_data_path = r"C:\Users\franc\OneDrive - Ngee Ann Polytechnic\EATC\ASSIGNMENT 2\for-2sec\for-2seconds" 
    
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    create_dataset_tensors(os.path.join(base_data_path, 'training'), os.path.join(output_dir, 'for_2_sec_train.pt'))
    create_dataset_tensors(os.path.join(base_data_path, 'validation'), os.path.join(output_dir, 'for_2_sec_valid.pt'))
    create_dataset_tensors(os.path.join(base_data_path, 'testing'), os.path.join(output_dir, 'for_2_sec_test.pt'))

    train_data = torch.load('data/for_2_sec_train.pt')
    print(train_data['data'].shape, train_data['labels'].shape)