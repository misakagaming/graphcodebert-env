from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path

paths = [str(x) for x in Path(r"C:/Users/eraye/OneDrive/Masaüstü/environment/containers/graphcodebert-env/gen2java").glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(r"C:\Users\eraye\OneDrive\Masaüstü\environment\containers\graphcodebert-env\translation\tokenizer")