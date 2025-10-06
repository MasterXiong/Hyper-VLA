import jax
import numpy as np

from octo.utils.train_utils import hf_weights_loader
from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.tokenizers import LanguageTokenizer


def get_language_tokenizer(tokenizer_kwargs):
    tokenizer = HFTokenizer(
        **tokenizer_kwargs
    )

    token_embedding_model = LanguageTokenizer('t5-base')
    rng = jax.random.PRNGKey(42)
    tasks = {'language_instruction': {'input_ids': np.ones((1, 16), dtype=np.int64), 'attention_mask': np.ones((1, 16), dtype=np.int64)}}
    params = token_embedding_model.init(rng, dict(), tasks, train=True)['params']
    # Load pretrained weights
    params = hf_weights_loader(params, hf_model="t5-base")

    return tokenizer, token_embedding_model, params


def token_to_embedding(model, params, tokens):
    tasks = {'language_instruction': tokens}
    embedding = model.apply({'params': params}, dict(), tasks, train=True)
    embedding = np.array(embedding.tokens)
    return embedding
