# Day 3: Transformer Architecture

Walking through how a transformer processeses one real example from our dataset step by step

**Example headline:** "Why Salesforce Stock Rallied Today"

##Why not keep using LSTM
Back on Week 3 Day 6, the LSTM read this headline one word at a time, left to right, keeping a running memory as it went. Two big problems with doing it that way:

It's slow because it cannot skip ahead. To get to word 5 ("Today"), the model has to fully process words 1 through 4 first, in order. Nothing happens in parallel, every step waits on the one before it. Fine for a 5-word headline, painful for anything long.

It forgets stuff. LSTMs have gates built specifically to hold onto important info longer, but early words still fade as they get pushed through step after step after step. By the end of a long paragraph, the model's memory of the first sentence is pretty blurry.

Transformers just don't do the one-at-a-time thing. Every word gets processed at the same time, and instead of passing memory word-to-word, each word looks directly at every other word in the sentence all at once. That "look at everything at once" move is called self attention, and basically the whole rest of the architecture is built around it.

## From words to vectors

Same first steps as Day 1 and Day 2, nothing new here:

1. Tokenize. Chop the headline into pieces: ["why", "salesforce", "stock", "rallied", "today"]. Real transformers use subword tokenization, so a rare word like "Salesforce" might become ["sales", "force"], which lets the model handle words it's never seen by building them from familiar chunks.
2. Embed. Every token gets swapped for a vector, exactly like the GloVe vectors from Day 2. Similar words start out with similar vectors.

Then a problem shows up: the model has no idea what order the words are in. A transformer looks at all words at once, there's no left-to-right reading, so "Why Salesforce Stock Rallied Today" and "Today Rallied Stock Salesforce Why" look identical to it.

Positional encoding fixes this. Before the words go into attention, the model adds a second vector to each word's embedding, one that encodes where the word sits in the sentence (position 1, 2, 3...). So "stock" at position 3 gets a slightly different final vector than the same word at position 1. Now the model can tell order apart even though it's still processing everything simultaneously. Positional encoding basically only exists because self-attention threw away word order, it's a patch that hands the order info back as a separate signal.

## Self-attention, the actual core

The whole job of self-attention: for each word, work out which other words in the sentence are relevant to it, then pull in info from those words.

Take "rallied" in our headline. On its own it's vague, rallied how? what rallied? Self-attention lets "rallied" look around and basically go: "stock is super relevant to me, salesforce tells me whose stock, why is asking about my cause, today is the when." It then builds a new version of the "rallied" vector that's blended with info from those relevant words. Now it's not a generic word anymore, it carries "Salesforce's stock, today" baked into it.

The mechanism from the Alammar article: every word makes three vectors out of its embedding.

- Query: here's what I'm looking for
- Key: here's what I'm about
- Value: here's the actual info I hand over if you pick me

For "rallied" to do its thing: take its Query and compare it against every other word's Key (a dot product, which is just a similarity score). High score means that word is relevant to "rallied." Run all those scores through softmax so they add up to 1 (same softmax from Week 3), which turns them into weights. Then take a weighted average of every word's Value vector using those weights. That weighted average is the new "rallied" vector.

Every word does this at the same time, it's all matrix multiplication under the hood, which is exactly why it parallelizes and the LSTM couldn't.

A library analogy makes the Q/K/V names click: Query is your search request, Keys are the labels on book spines you scan, Values are the actual books you pull off the shelf. Compare your request to every label, then grab content from the best matches, weighted by how good each match was.

## Multi-head attention

One round of self-attention gives every word one new context-blended vector. Multi-head attention runs several of those rounds in parallel, each with its own separate Q/K/V weights, so each "head" can learn to focus on a different kind of relationship.

For "rallied" you might get something like:

- Head 1 learns to lock onto the subject: "stock" and "salesforce" (what rallied)
- Head 2 learns to lock onto question/cause words: "why"
- Head 3 learns to lock onto time: "today"

Each head produces its own version of the new "rallied" vector. They all get concatenated together and passed through one more small linear layer that mixes them back down into a single vector. So "rallied" comes out having captured several different types of context at once instead of just one.

Nothing new mechanically, it's the exact same Q/K/V attention from the section above, just run 8 or 12 times side by side with different learned weights, then combined.

## The transformer block, encoder, decoder

Stack these pieces and you get one "block":

1. Multi-head self-attention
2. Add the block's input back onto its output (a residual connection, keeps the original signal from getting lost) and normalize
3. A small plain feed-forward network applied to each word
4. Another residual + normalize

Real transformers stack this block many times. BERT-base uses 12, bigger models use dozens. Each block lets every word gather more context, and stacking them lets later blocks build on the understanding earlier ones produced.

Encoder vs decoder:

- Encoder: stacks of the block above. Its job is to build a rich, context-aware representation of the whole input. Every word sees every other word, no restrictions. This is what BERT/RoBERTa are (Day 4).
- Decoder: similar, but with two changes. Each word can only attend to words before it (so it can't cheat by seeing the future while generating text left to right), plus an extra attention layer that looks back at the encoder's output. This is what GPT-style models use (decoder only, no encoder).

For our headline classification task, we only need the encoder: build a representation of "Why Salesforce Stock Rallied Today," then feed that into a classifier. No text generation, so no decoder needed.

## Recap: the full flow for our headline

1. "Why Salesforce Stock Rallied Today" gets tokenized into pieces
2. Each token becomes an embedding vector
3. Positional encoding is added so word order isn't lost
4. Multi-head self-attention: every word looks at every other word at once, and gets rewritten with context from whichever words mattered to it
5. Residual connection + normalize + a small feed-forward network
6. Repeat steps 4-5 for however many blocks the model has
7. The final vectors are context-rich representations of each word; pool or take a summary vector and feed it into a classifier for the category

The one big shift from the LSTM: no sequential chain, no fading memory. Every word gets direct access to every other word in a single parallel step, and stacking blocks deepens the understanding instead of stretching a fragile memory across more timesteps.

