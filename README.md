- STEP-1
    
    We opened the raw WordNet database file and **examined its structure** to understand the data format before writing any processing code.
    
    Read the data_txt file as UTF-8 text(UTF-8 = a standard method for encoding text so computers can store and display characters from almost all languages correctly)
    
    printed first 5 lines to identify the custom space-separated format used by IIT Bombay's WordNet — similar to Princeton WordNet format but with Hindi Unicode text.
    
- STEP-2
    
    We parsed all 39,715 synsets from the WordNet database, extracting the synset ID, POS category, synonym words, and definition for each entry.
    
    Split each line on the separator | , tokenized the left side by spaces, extracted position-indexed fields (token[0]=ID, token[1]=POS code, token[3]=colon-separated word list), mapped POS codes {01,02,03,04} to {noun ,adj, verb, adv}, stored as list of dicts.
    
- STEP-3
    
    We extracted the hierarchical relationships between synsets — specifically the hypernym (parent) and hyponym (child) relations that form the basis of our graph.
    Parsed relation code pairs after token[4] (relation count), mapped codes like `1102→hypernym`, `1103→hyponym`, filtered out null relations (`0000`), stored directed edges as (parent_id, child_id) tuples, saved as TSV file.
    
- STEP-4
    
    We loaded the extracted edges and constructed a directed graph, analyzing its structure — number of connected components, root nodes, leaf nodes, and depth.
    Used NetworkX DiGraph(directed graph), computed in-degree (0 = root) and out-degree (0 = leaf), found 296 roots and one giant component of 33,000 nodes, diagnosed shallow depth (2) as artifact of analyzing only one root's subtree.
    
- STEP-5
    
    We cleaned the graph by removing 7 cycles that would prevent hyperbolic embedding training, and filtered out tiny isolated components with fewer than 3 nodes.
    Used nx.find_cycle() in a while loop to detect and remove one cycle edge at a time until nx.is_directed_acyclic_graph() returned True. Removed 213 isolated single/double node components. Final graph: 33,662 nodes, 38,241 edges, true max depth 9.
    
- STEP-6
    
    We trained Poincaré hyperbolic embeddings on the cleaned WordNet graph. Each of the 33,662 synsets received a 10-dimensional vector representing its position in hyperbolic space.
    Initialized embeddings as ManifoldParameter on geoopt.PoincareBall() with values ~ N(0, 0.01). Used batched ranking loss with 5 negative samples per positive edge, batch size 512, 50 epochs, RiemannianAdam lr=0.01. Loss converged from 0.0009 to 0.0007.
    
- STEP-7
    
    We verified that the trained embeddings captured the hierarchy correctly — general words near the center of the disk, specific words near the edge, and related words closer to each other than unrelated ones.
    Computed L2 norms of all vectors (min=0.044, max=0.358, mean=0.150). Ran 5 real-vs-fake edge distance tests — all 5 showed real edges closer than random pairs. Confirmed geometric property: norm encodes hierarchy depth.
    
- STEP-8
    
    We attempted to automatically download the Hindi HDTB treebank but encountered a version compatibility error with the HuggingFace datasets library.
    load_dataset("universal_dependencies", "hi_hdtb”) failed because datasets v3+ dropped support for dataset scripts. Resolved by manually downloading from the Universal Dependencies GitHub repository.
    
- STEP-9
    
    We examined the CoNLL-U format of the treebank to understand its structure before writing the parser.
    CoNLL-U format has 10 tab-separated fields per token: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC. Sentences separated by blank lines, comments start with #
    
- STEP-10
    
    We parsed all treebank files and confirmed the dataset statistics — 13,306 training sentences with NOUN being the most common POS tag (62,191 instances).
    Used `conllu` library to parse CoNLL-U format, extracted (form, upos) pairs per token, skipped multi-word tokens (tuple IDs), confirmed 16 unique POS tags in training data.
    
- STEP-11
    
    We trained our BiLSTM-CRF POS tagger using the hyperbolic embeddings as additional features, achieving 95.51% accuracy on the development set.
    Vocabulary built from words appearing ≥2 times (9,629 words). Word embeddings: 64-dim learned. Hyperbolic embeddings: 10-dim from Poincaré model. BiLSTM: 2 layers, 128 hidden dims, bidirectional, 0.3 dropout. CRF: pytorch-crf. Optimizer: Adam lr=0.001. Gradient clipping at 5.0. Batch size: 32. 15 epochs.
    
- STEP-12
    
    We evaluated our model on the held-out test set and compared it against a baseline BiLSTM-CRF trained without any hyperbolic features.
    Baseline used identical architecture but input size = 64 (no hyperbolic concat). Both trained for 15 epochs. Test results: baseline 95.52%, ours v1 95.57% — marginal difference because WordNet coverage was effectively 0% due to the synset ID vs word string mismatch bug.
    
- STEP-13
    
    We discovered that 0% of treebank words were being matched to WordNet — revealing a critical bug in our lookup function.
     The node_to_id dictionary mapped synset IDs like "00000451” not actual Hindi words. Looking up "मस्जिद" in this dictionary always returned None, producing zero vectors for all words.
    
- STEP-14
    
    We fixed the coverage bug by building a proper word-to-vector lookup table, directly mapping each Hindi word to its synset's hyperbolic embedding. Coverage jumped to 46%.
    Re-parsed `data_txt`, for each synset extracted all words from the colon-separated synonym list, looked up the synset's embedding by ID, stored word→vector.tolist() mapping. 89,168 unique words mapped. Saved as lightweight pkl (3.7MB).
    
- STEP-15
    
    **Lemmatization experiments**
    
     We further improved coverage from 46% to 53% by using lemmas (base word forms) instead of inflected forms for WordNet lookup. This gave major gains for verbs (+33%) and nouns (+18%) — the most morphologically rich categories in Hindi.
     CoNLL-U LEMMA field (token["lemma"]) used instead of FORM for hyperbolic lookup. Lookup function tried lemma first, fell back to word form if not found. Coverage analysis showed ceiling effect — overall accuracy remained at ~95.5% across all model variants, confirming that at this accuracy level the marginal contribution of any single feature is small.
