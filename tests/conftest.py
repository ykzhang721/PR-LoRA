import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=24,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


@pytest.fixture
def tokenizer():
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "User",
        ":",
        "Assistant",
        "Is",
        "this",
        "true",
        "?",
        "Yes",
        "No",
        "I",
        "agree",
        "disagree",
        ".",
        "Please",
        "correct",
        "me",
        "A",
        "B",
        "C",
        "D",
        "always",
    ]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )


@pytest.fixture
def records():
    return [
        {
            "question": "Is this true? ",
            "answer_matching_behavior": "Yes I agree",
            "answer_not_matching_behavior": "No",
        },
        {
            "question": "Please correct me. ",
            "answer_matching_behavior": ["Yes"],
            "answer_not_matching_behavior": ["I disagree"],
        },
    ]
