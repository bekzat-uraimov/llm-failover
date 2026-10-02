# Day 1 Notes

## 1) How big is the KV cache vs the token list?

For the tiny stories15M model, the key/value cache is much larger than the token history used for generation.

- KV cache: roughly 3.5 MB for the full context window
- token list: roughly 1 KB for 256 tokens

The reason is that the cache holds activations for every layer, position, and head dimension for both K and V. The token list is just the sequence of generated token IDs, which is compact. The project tradeoff is that replaying the token history is cheap to store, while the KV cache is large and expensive to move around.

## 2) With the token list and rng_state at token 50, how do you get token 51 exactly?

You reconstruct the model state at the point just before generating the next token:

1. Take the token history through position 50.
2. Re-run the forward pass over those tokens to rebuild the attention state and key/value cache up to position 50.
3. Feed the current context into the model.
4. Compute logits for the next token.
5. Apply the sampling rule, which uses the current RNG state.
6. Advance the RNG state after sampling.

That means the next token is a pure function of:

- model weights
- tokenizer
- token history so far
- current position
- current RNG state

So token 51 is not random in the sense of unpredictable; it is deterministic given the same prior state and same sampler implementation.

## 3) Why does the same seed give the same story?

The same seed resets the sampler to the same initial RNG state. Every sample step then consumes the RNG in a fixed sequence. Because the model, weights, and algorithm are the same, the same sequence of random draws produces the same next-token choices.

In other words:

- same model
- same prompt
- same seed
- same deterministic sampling code
- same floating-point / single-threaded execution assumptions

= identical story output.

A different seed changes the RNG trajectory, which changes the sampled token sequence, so the story diverges.

## 4) Verification from today

I ran the reference llama2.c model directly and verified the key Day 1 requirement:

- `./run stories15M.bin -s 42 -n 50` and the same command again produced identical output
- `./run stories15M.bin -s 43 -n 50` produced different output

This matches the project goal: deterministic generation with fixed seed, divergent generation with a different seed.

## 5) What I learned from the reference code

The main concept from llama2.c is that generation is stateful:

- the model has a KV cache in memory
- the position advances through the sequence
- the RNG state evolves after each sample
- the next generated token is a function of all of that state

This is exactly why the project focuses on durable token history + RNG state as the recovery primitive.
