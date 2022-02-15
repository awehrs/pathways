import jax
import jax.numpy as jnp
from jaxline import base_config
from ml_collections import config_dict
import numpy as np
import optax
import os
import sys
import haiku as hk

sys.path.append("..")

print(sys.path)
print(os.pardir)
from train import dataset
import io_processors
import perceiver
import configs.multi_pathway_single_head as c1
import configs.multi_pathway_multi_head as c2
import configs.single_pathway_multihead as c3

N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples
N_CLASSES = 10
IS_LOCAL = True

config = c1.get_config()


def _forward_fn(
    inputs: dataset.Batch,
) -> jnp.ndarray:

    images = inputs["images"]

    perceiver_kwargs = config.experiment_kwargs.config.model.perceiver_kwargs
    input_preprocessor = io_processors.ImagePreprocessor(
        **perceiver_kwargs["input_preprocessor"]
    )
    encoder = perceiver.PerceiverEncoder(**perceiver_kwargs["encoder"])
    decoder = perceiver.ClassificationDecoder(10, **perceiver_kwargs["decoder"])
    model = perceiver.Perceiver(
        encoder=encoder, decoder=decoder, input_preprocessor=input_preprocessor
    )

    return model(images, is_training=True)


split = dataset.Split.TRAIN_AND_VALID
batch_dims = [64]
ds = dataset.load(
    split=split,
    is_training=True,
    batch_dims=batch_dims,
    im_dim=config.experiment_kwargs.config.data.im_dim,
    augmentation_settings=config.experiment_kwargs.config.data.augmentation,
)
inputs = next(ds)
forward = hk.transform(_forward_fn)
rng_key = jax.random.PRNGKey(42)
params = forward.init(rng=rng_key, inputs=inputs)
# print(hk.experimental.tabulate(forward)(inputs))
x1 = next(ds)
x2 = next(ds)
print(jax.nn.softmax(forward.apply(params, rng_key, x1))[0])
print()
