import argparse

from sklearn.model_selection import train_test_split

from src.train import get_device, TactileDataset, combine_data_dicts, preprocess_combined_data
from src.model import ModelFactory

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from collections import defaultdict
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from nltk.corpus import brown, gutenberg
import json
import pickle
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


def load_sample_text() -> str:
    """
    Load a sample folktale text.
    Using 'The Three Little Pigs' as an example.
    """
    text = """
Once upon a time there were three little pigs. The time came for them to leave home and seek their fortunes.
Before they left, their mother told them whatever you do, do it the best that you can because that is the way to get along in the world.
The first little pig built his house out of straw because it was the easiest thing to do.
The second little pig built his house out of sticks. This was a little bit stronger than a straw house.
The third little pig built his house out of bricks. This was the strongest house of all.
One day the big bad wolf came along and saw the first little pig in his house of straw.
He said little pig, little pig, let me come in. But the little pig said no, not by the hair on my chinny chin chin.
Then I will huff, and I will puff, and I will blow your house in. So he huffed, and he puffed, and he blew the house in.
The little pig ran to the second pig's house that was made of sticks.
The big bad wolf came to that house and said little pig, little pig, let me come in.
But the pig said no, not by the hair on my chinny chin chin.
Then I will huff, and I will puff, and I will blow your house in.
So he huffed, and he puffed, and he blew the house down.
The two little pigs ran to the third pig's house that was made of bricks.
The big bad wolf came to the house and said little pig, little pig, let me come in.
But the pig said no, not by the hair on my chinny chin chin.
Then I will huff, and I will puff, and I will blow your house in.
Well, the wolf huffed and puffed but he could not blow the brick house down.
The wolf was very angry and decided to come down the chimney.
The three little pigs lit a fire in the fireplace and put on a pot of water.
When the wolf came down the chimney, he landed in the pot of hot water.
The wolf jumped out of the pot and ran away. He never came back to bother the three little pigs again.
The three little pigs lived happily ever after in the brick house.
"""
    # Clean the text
    return ' '.join(text.lower().split())


def create_sentence_dataset(test_dataset, idx_to_char: Dict[int, str]) -> List[Tuple[torch.Tensor, str]]:
    """Create evaluation dataset using a folktale."""
    # Create mapping of letters to their available tactile samples
    letter_samples = defaultdict(list)
    for inputs, label in test_dataset:
        letter = idx_to_char[label.item()]
        letter_samples[letter].append(inputs)

    # Load and prepare text
    text = load_sample_text()

    # Create dataset
    sentence_data = []
    skipped_chars = set()

    for char in text:
        if char in letter_samples:
            # Randomly select a tactile sample for this character
            random_index = np.random.randint(len(letter_samples[char]))
            sample = letter_samples[char][random_index]
            sentence_data.append((sample, char))
        elif char not in ['\n', '\t', '\r']:  # Don't log whitespace
            skipped_chars.add(char)

    # Report any characters that were skipped
    if skipped_chars:
        print(f"Warning: The following characters were not in the tactile dataset: {sorted(skipped_chars)}")

    print(f"Created dataset with {len(sentence_data)} characters")

    return sentence_data


class NLTKLanguageModel:
    def __init__(self, n: int = 4, model_path: str = 'char_lm.pkl'):
        """Initialize NLTK language model with pre-trained data."""
        self.n = n
        self.model_path = model_path

        if os.path.exists(model_path):
            print(f"Loading pre-trained language model from {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("No pre-trained model found. Training new language model...")
            self.model = KneserNeyInterpolated(n)
            self._ensure_nltk_data()
            self._train_model()
            self._save_model()

    def _save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('corpora/brown')
            nltk.data.find('corpora/gutenberg')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('brown')
            nltk.download('gutenberg')

    def _train_model(self):
        """Train the language model on Brown and Gutenberg corpora."""
        print("Loading and preprocessing training data...")

        texts = []
        print("Processing Brown corpus...")
        texts.extend([
            ' '.join(sent).lower()
            for sent in brown.sents()
        ])

        print("Processing Gutenberg corpus...")
        texts.extend([
            ' '.join(sent).lower()
            for sent in gutenberg.sents()
        ])

        print("Creating character sequences...")
        char_sequences = []
        for text in texts:
            sequence = list(text)  # This will naturally include spaces, commas, etc.
            char_sequences.append(sequence)

        print("Creating n-grams...")
        train_data, vocab = padded_everygram_pipeline(self.n, char_sequences)

        print("Training language model...")
        self.model.fit(train_data, vocab)
        print("Language model training complete")

    def get_probabilities(self, context: str, tactile_char_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Get probability distribution over next characters, mapping to tactile model indices.
        """
        context = list(context.lower())
        probs = np.zeros(len(tactile_char_to_idx))

        # Create remapped version where 'space' is mapped to ' '
        lm_char_to_idx = tactile_char_to_idx

        # Get probabilities for all characters
        for char, idx in lm_char_to_idx.items():
            if char not in ['backspace', 'shift']:  # Skip special function keys
                try:
                    prob = self.model.score(char, context[-(self.n - 1):])
                    probs[idx] = prob
                except ValueError:
                    probs[idx] = 1e-10

        # Add very low probabilities for special function keys
        probs[tactile_char_to_idx['backspace']] = 1e-10
        probs[tactile_char_to_idx['shift']] = 1e-10

        # Normalize to get probability distribution
        probs = probs / (probs.sum() + 1e-10)
        return probs


class PretrainedLanguageModel:
    def __init__(self, model_name: str = 'gpt2'):
        """Initialize with a pre-trained GPT-2 model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading pre-trained GPT-2 model ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_probabilities(self, context: str, tactile_char_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Get probability distribution over next characters, mapping to tactile model indices.
        """
        # Prepare input
        context = context.lower()
        inputs = self.tokenizer(context, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Get logits for next token prediction

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Initialize output probability array
        final_probs = np.zeros(len(tactile_char_to_idx))

        # Map token probabilities to character probabilities
        for char, idx in tactile_char_to_idx.items():
            if char not in ['backspace', 'shift']:
                # Get all tokens that start with this character
                char_tokens = [i for i, token in enumerate(self.tokenizer.get_vocab().keys())
                               if token.startswith('Ġ' + char) or token.startswith(char)]

                if char_tokens:
                    # Sum probabilities of all tokens starting with this character
                    char_prob = probs[0, char_tokens].sum().item()
                    final_probs[idx] = char_prob
                else:
                    final_probs[idx] = 1e-10
            else:
                # Special function keys get very low probability
                final_probs[idx] = 1e-10

        # Normalize to get probability distribution
        final_probs = final_probs / (final_probs.sum() + 1e-10)
        return final_probs

    def __call__(self, context: str, tactile_char_to_idx: Dict[str, int]) -> np.ndarray:
        """Convenience method to call get_probabilities directly."""
        return self.get_probabilities(context, tactile_char_to_idx)


class GPT2CharacterLM:
    def __init__(self):
        """Initialize GPT-2 with character-level tokenizer."""
        print("Loading GPT-2 model and creating character tokenizer")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load base model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        base_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        # Create character vocabulary
        chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # a-z
        chars.extend([' ', ','])  # Add space and comma

        # Train new tokenizer with character vocabulary
        self.tokenizer = base_tokenizer.train_new_from_iterator(
            [[c] for c in chars],  # Each character as a separate sequence
            vocab_size=len(chars),
            initial_alphabet=chars
        )

        self.model.eval()

    def get_probabilities(self, context: str, char_to_idx: Dict[str, int]) -> np.ndarray:
        """Get probability distribution over next characters."""
        probs = np.zeros(len(char_to_idx))

        with torch.no_grad():
            # Tokenize context
            inputs = self.tokenizer(context, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model predictions
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Get logits for next token prediction

            # Convert to probabilities
            char_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Map character probabilities to our index space
            for char, idx in char_to_idx.items():
                if char == 'backspace' or char == 'shift':
                    probs[idx] = 1e-10
                elif char == 'space':
                    token_id = self.tokenizer.encode(' ')[0]
                    probs[idx] = char_probs[token_id]
                else:
                    try:
                        token_id = self.tokenizer.encode(char)[0]
                        probs[idx] = char_probs[token_id]
                    except:
                        probs[idx] = 1e-10

            # Normalize probabilities
            probs = probs / (probs.sum() + 1e-10)

        return probs


def evaluate_pure_tactile(
        model: torch.nn.Module,
        sentence_data: List[Tuple[torch.Tensor, str]],
        device: torch.device,
        idx_to_char: Dict[int, str]
) -> Tuple[float, List[str]]:
    """Evaluate model without language model assistance."""
    model.eval()
    correct = 0
    total = 0
    predicted_text = []

    # Process in batches for GPU efficiency
    batch_size = 64
    current_batch = []

    with torch.no_grad():
        for i, (inputs, true_char) in enumerate(sentence_data):
            current_batch.append((inputs, true_char))

            if len(current_batch) == batch_size or i == len(sentence_data) - 1:
                # Process batch
                batch_inputs = torch.stack([item[0] for item in current_batch]).to(device)
                batch_true = [item[1] for item in current_batch]

                outputs = model(batch_inputs)
                predicted_idx = outputs.argmax(1)

                for idx, true_char in zip(predicted_idx, batch_true):
                    predicted_char = idx_to_char[idx.item()]
                    correct += (predicted_char == true_char)
                    total += 1
                    predicted_text.append(predicted_char)

                current_batch = []

    accuracy = 100. * correct / total
    return accuracy, predicted_text


def evaluate_with_language_model(
        model: torch.nn.Module,
        sentence_data: List[Tuple[torch.Tensor, str]],
        idx_to_char: Dict[int, str],
        char_to_idx: Dict[str, int],
        lm: NLTKLanguageModel,
        device: torch.device,
        alpha: float = 0.7,
) -> Tuple[float, List[str], Dict]:
    """Evaluate model with language model assistance."""
    model.eval()
    correct = 0
    total = 0
    current_context = '____'
    predicted_text = []
    analysis = {
        'correct_both': 0,
        'correct_tactile_only': 0,
        'correct_lm_only': 0,
        'wrong_both': 0,
        'predictions': []
    }

    # Process each character sequentially
    with torch.no_grad():
        for i, (inputs, true_char) in enumerate(sentence_data):
            # Get tactile model prediction
            inputs = inputs.unsqueeze(0).to(device)
            outputs = model(inputs)
            tactile_probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

            # Get language model prediction
            lm_probs = lm.get_probabilities(current_context, char_to_idx)

            # Get individual predictions
            tactile_pred = idx_to_char[np.argmax(tactile_probs)]
            lm_pred = idx_to_char[np.argmax(lm_probs)]

            # Combine predictions
            combined_probs = alpha * tactile_probs + (1 - alpha) * lm_probs
            predicted_idx = np.argmax(combined_probs)
            predicted_char = idx_to_char[predicted_idx]

            # Update statistics
            correct += (predicted_char == true_char)
            total += 1

            # Record prediction analysis
            tactile_correct = (tactile_pred == true_char)
            lm_correct = (lm_pred == true_char)

            if tactile_correct and lm_correct:
                analysis['correct_both'] += 1
            elif tactile_correct:
                analysis['correct_tactile_only'] += 1
            elif lm_correct:
                analysis['correct_lm_only'] += 1
            else:
                analysis['wrong_both'] += 1

            analysis['predictions'].append({
                'true': true_char,
                'tactile': tactile_pred,
                'lm': lm_pred,
                'combined': predicted_char,
                'context': current_context
            })

            # Update context for next prediction
            if predicted_char not in ['backspace', 'shift']:
                context_char = ' ' if true_char == 'space' else true_char
                if isinstance(lm, NLTKLanguageModel):
                    current_context = current_context[1:] + context_char
                else:
                    current_context = current_context + context_char
                    if len(current_context) > 100:
                        current_context = current_context[-100:]
            predicted_text.append(predicted_char)

            # Print progress every 100 characters
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(sentence_data)} characters. "
                      f"Current accuracy: {100. * correct / total:.2f}%")

    accuracy = 100. * correct / total
    return accuracy, predicted_text, analysis


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # Load character mapping from JSON
    with open('class_mapping.json', 'r') as f:
        label_to_idx = json.load(f)
    idx_to_char = {int(v): k for k, v in label_to_idx.items()}
    # convert 'space' to ' '
    idx_to_char[label_to_idx['space']] = ' '

    data_folder = 'recordings'
    file_list = [f for f in os.listdir(data_folder)
                 if f.endswith('layout_1.hdf5') or f.endswith('2024-11-25_18-13-21.hdf5')]

    combined_data = combine_data_dicts(file_list, data_folder)
    samples, labels, label_to_idx = preprocess_combined_data(combined_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.3, random_state=42, stratify=labels
    )
    test_dataset = TactileDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    if 'resnet' in args.model_path.lower():
        model = ModelFactory.create_model('resnet', num_classes=30, device=device)
        model.load_state_dict(torch.load('models/best_ResNet_model.pth', map_location=device))
    elif 'knn' in args.model_path.lower():
        model = ModelFactory.create_model('knn', num_classes=30, device=device)
        train_dataset = TactileDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        model.fit(train_dataset)
    elif 'shallowcnn' in args.model_path.lower():
        model = ModelFactory.create_model('shallow_cnn', num_classes=30, device=device)
        model.load_state_dict(torch.load('models/best_ShallowCNN_model.pth', map_location=device))

    # Create sentence dataset from your test data
    sentence_data = create_sentence_dataset(test_dataset, idx_to_char)

    # First evaluate without language model
    print("\nEvaluating pure tactile prediction...")
    tactile_accuracy, tactile_text = evaluate_pure_tactile(
        model, sentence_data, device, idx_to_char
    )
    print(f"Pure tactile accuracy: {tactile_accuracy:.2f}%")

    # Initialize language model (will load from file if exists)
    print("\nInitializing language model...")
    lm = PretrainedLanguageModel()

    # Test different weightings with language model
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_combined_accuracy = 0
    best_alpha = 0

    print("\nEvaluating with language model assistance...")
    for alpha in alphas:
        print(f"\nTesting with alpha={alpha}")
        accuracy, predicted_text, analysis = evaluate_with_language_model(
            model, sentence_data, idx_to_char, label_to_idx, lm, device, alpha
        )

        if accuracy > best_combined_accuracy:
            best_combined_accuracy = accuracy
            best_alpha = alpha

        print(f"Combined accuracy: {accuracy:.2f}%")
        print("\nPrediction Analysis:")
        print(f"Both models correct: {analysis['correct_both']}")
        print(f"Only tactile correct: {analysis['correct_tactile_only']}")
        print(f"Only LM correct: {analysis['correct_lm_only']}")
        print(f"Both wrong: {analysis['wrong_both']}")

    # Final comparison
    print("\nFinal Results:")
    print(f"Pure tactile accuracy: {tactile_accuracy:.2f}%")
    print(f"Best combined accuracy: {best_combined_accuracy:.2f}% (alpha={best_alpha})")
    print(f"Absolute improvement: {best_combined_accuracy - tactile_accuracy:.2f}%")
    print(f"Relative improvement: {((best_combined_accuracy / tactile_accuracy) - 1) * 100:.2f}%")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, default='models/best_ShallowCNN_model.pth',
                           help='Path to the best model checkpoint')
    args = argparser.parse_args()
    main(args)
