# File Descriptions

This document provides a directory-wise listing of the Python files in this repository, categorized by their primary use case: **Training** or **Inference**.

---

## 1. Files Used Only for Training
These files are responsible for data preprocessing, dataset management, and model training. They are not required to run the pre-trained models.

### Root Directory
- **encoder_preprocess.py**: entry point for preprocessing audio data for the speaker encoder.
- **encoder_train.py**: entry point for training the speaker encoder model.
- **synthesizer_preprocess_audio.py**: entry point for preprocessing audio into mel spectrograms for the synthesizer.
- **synthesizer_preprocess_embeds.py**: entry point for generating speaker embeddings for the synthesizer training set.
- **synthesizer_train.py**: entry point for training the Tacotron synthesizer.
- **vocoder_preprocess.py**: entry point for preprocessing data for the vocoder training.
- **vocoder_train.py**: entry point for training the WaveRNN vocoder.

### encoder/
- **encoder/preprocess.py**: contains logic for cleaning and formatting audio for the encoder.
- **encoder/train.py**: implementation of the speaker encoder training loop and loss functions.
- **encoder/visualizations.py**: utilities for visualizing embeddings during training (e.g., UMAP plots).
- **encoder/data_objects/**:
    - **random_cycler.py**: utility for cycling through training data randomly.
    - **speaker_batch.py**: object representing a batch of utterances from multiple speakers.
    - **speaker_verification_dataset.py**: PyTorch dataset implementation for speaker verification.
    - **speaker.py**: represents a speaker and their associated utterances.
    - **utterance.py**: represents a single audio utterance.

### synthesizer/
- **synthesizer/preprocess.py**: logic for converting audio files into the format expected by the synthesizer.
- **synthesizer/synthesize.py**: generates ground-truth-aligned (GTA) mel spectrograms used for training the vocoder.
- **synthesizer/synthesizer_dataset.py**: PyTorch dataset implementation for the Tacotron synthesizer.
- **synthesizer/train.py**: implementation of the synthesizer training loop.

### vocoder/
- **vocoder/train.py**: implementation of the WaveRNN vocoder training loop.
- **vocoder/vocoder_dataset.py**: PyTorch dataset implementation for the vocoder.
- **vocoder/gen_wavernn.py**: utility to generate test samples from a partially trained vocoder for evaluation.
- **vocoder/distribution.py**: implementation of probability distributions (e.g., Discretized Mixture of Logistics) and loss functions for WaveRNN.
- **vocoder/models/deepmind_version.py**: alternative architecture for WaveRNN based on the DeepMind implementation.

---

## 2. Files Must-Have for Inference
These files are essential for running the pre-trained models to clone voices. They include the model architectures, inference logic, and the UI/CLI wrappers.

### Root Directory
- **demo_cli.py**: a command-line interface to test the voice cloning system end-to-end.
- **demo_toolbox.py**: the main entry point for the graphical user interface (GUI) toolbox.

### encoder/
- **encoder/inference.py**: high-level API for loading the encoder and generating speaker embeddings from audio.
- **encoder/model.py**: the speaker encoder (d-vector) model architecture.
- **encoder/audio.py**: audio processing utilities (normalization, mel filterbank) used by the encoder.
- **encoder/params_data.py**: constants and parameters for encoder audio processing.
- **encoder/params_model.py**: hyperparameters for the encoder model architecture.
- **encoder/config.py**: configuration management for the encoder.

### synthesizer/
- **synthesizer/inference.py**: high-level API for loading the synthesizer and generating mel spectrograms from text.
- **synthesizer/models/tacotron.py**: the Tacotron 2 model architecture.
- **synthesizer/audio.py**: audio and spectrogram processing utilities for the synthesizer.
- **synthesizer/hparams.py**: hyperparameters for the synthesizer model and audio processing.
- **synthesizer/utils/**:
    - **cleaners.py**: text normalization and cleaning (e.g., numbers to words).
    - **symbols.py**: the set of characters/phonemes supported by the synthesizer.
    - **text.py**: logic for converting raw text into sequences of symbol IDs.
    - **numbers.py**: utility for expanding numbers in text.
    - **plot.py**: utilities for plotting spectrograms and alignments.

### vocoder/
- **vocoder/inference.py**: high-level API for loading the vocoder and generating waveforms from spectrograms.
- **vocoder/models/fatchord_version.py**: the WaveRNN model architecture (standard version).
- **vocoder/audio.py**: audio processing utilities specifically for the vocoder.
- **vocoder/hparams.py**: hyperparameters for the vocoder model.
- **vocoder/display.py**: utilities for displaying progress bars and tables in the terminal.

### toolbox/
- **toolbox/ui.py**: implementation of the PyQt5-based graphical user interface.
- **toolbox/utterance.py**: data structure to represent an utterance within the toolbox.
- **toolbox/__init__.py**: contains the `Toolbox` class that orchestrates the three models and the UI.

### utils/
- **utils/argutils.py**: utilities for parsing and printing command-line arguments.
- **utils/default_models.py**: logic for ensuring pre-trained models are downloaded and available.
- **utils/logmmse.py**: algorithm for noise reduction, used in audio preprocessing for better cloning quality.
- **utils/profiler.py**: simple profiling utility for measuring execution time.
