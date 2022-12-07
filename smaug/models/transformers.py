import transformers

from typing import Tuple


def mT5_load(
    model_name: str = "google/mt5-large",
) -> Tuple[transformers.MT5ForConditionalGeneration, transformers.T5Tokenizer]:
    """Loads mT5 model and tokenizer.

    Args:
        model_name (str, optional): name of the mT5 model to use. Defaults to "google/mt5-large".

    Returns:
        Tuple[transformers.MT5ForConditionalGeneration, transformers.T5Tokenizer]: mT5 model and tokenizer.
    """
    model = transformers.MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


POLYJUICE_EOF_TOKEN = "<|endoftext|>"


def polyjuice_load():
    """Loads PolyJuice model for constrained generation.

    Returns:
        Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]: PolyJuice model and tokenizer.
    """
    model_path = "uw-hai/polyjuice"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        pad_token=POLYJUICE_EOF_TOKEN,
    )
    return model, tokenizer


def roberta_mnli_load() -> Tuple[
    transformers.AutoModelForSequenceClassification, transformers.AutoTokenizer
]:
    """Loads RoBERTa finetuned for multilingual natural language inference.

    Returns:
        Tuple[transformers.AutoModelForSequenceClassification, transformers.AutoTokenizer]: RoBERTa model and tokenizer.
    """
    name = "roberta-large-mnli"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer
