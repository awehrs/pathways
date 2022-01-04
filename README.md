# Pathway Perceiver

A simple tweak to the architecture of Deepmind's 
Perceiver [1] and subsequent PerceiverIO [2], 
with code forked from Deempind's Haiku implementation
of same:
https://github.com/deepmind/deepmind-research/blob/master/perceiver
(c) Deepmind

This architecture partitions the latent arrays of the Perceiver's 
encoder into multiple parallel pathways, each comprised of
an equal number of latents. This change accomplishes two things:
1) reduces the complexity of each self-attend by an order of N, 
where N is the number of parallel pathways. 
2) allows for deeper exploration of representation subspaces than 
afforded by a purely multiheaded approach--likely a valuable feature
for multimodal data. 

Currently, pathway fusion occurs at the decoding stage. Future 
iterations will likely perform fusion with sparse cross attention 
between pathways. 

## Usage

After installing dependencies with `pip install -f requirements.txt`,
re-install ml_collections from source: 
`pip install git+https://github.com/google/ml_collections`
(pip's version of ml_collecitons is too old to play nice with Jaxline)


## Attributions and Disclaimers

Except for pathway-related architectural adjustments in 
perceiver.py and position_encoding.py (and related adjustments
necessiated in experimental configurations) all code is 
taken from https://github.com/deepmind/deepmind-research/blob/master/perceiver
(c) Deepmind

## References

[1] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals,
João Carreira.
*Perceiver: General Perception with Iterative Attention*. ICML 2021.
https://arxiv.org/abs/2103.03206

[2] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch,
Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock,
Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman,
Oriol Vinyals, João Carreira.
*Perceiver IO: A General Architecture for Structured Inputs & Outputs*.
arXiv, 2021.
https://arxiv.org/abs/2107.14795
