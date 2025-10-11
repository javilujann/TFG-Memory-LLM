# Research Notes

## Literature Review

### Survey Memoria LLM
- Main survey paper on LLM memory mechanisms
- Key findings: [Add your notes here]

### Datasets

#### LongMemEval
- **NEW** benchmark (2024) for evaluating long-term memory in chat assistants
- **Core abilities tested**: Information extraction, multi-session reasoning, temporal reasoning, knowledge updates, conversational synthesis
- **Focus**: Sustained interactions over multiple sessions, personalized responses
- **Key challenge**: Evaluating how well LLMs maintain and use information across long chat histories
- **Published**: ICLR 2025 (very recent!)
- **Importance for your project**: Direct evaluation of memory capabilities in interactive scenarios

#### NarrativeQA
- **Established** dataset (2017) for reading comprehension on long narratives
- **Task**: Question answering about books and movie scripts
- **Key challenge**: Requires integrating information across full documents, not just local extraction
- **Focus**: Multi-hop reasoning, narrative understanding, character tracking
- **Document length**: Entire books/scripts (very long context)
- **State-of-the-art applications**: Used extensively for testing long-context models, memory networks, and retrieval-augmented approaches
- **Importance for your project**: Gold standard for testing memory on narrative understanding tasks

#### BookSum
- Book summarization dataset
- Challenges: Long-context understanding
- Notes: [Add your analysis here]

## Ideas & Hypotheses

### Memory Systems to Compare
1. **Baseline**: Standard Transformer
2. **Retrieval-Augmented**: RAG-style approaches
3. **Attention-based**: Extended attention mechanisms
4. **External Memory**: Memory-augmented networks
5. **Temporal Memory**: Systems with explicit temporal reasoning (relevant for LongMemEval)
6. **Session-based Memory**: Multi-session conversation memory

### Key Research Insights from Literature

#### Recent Advances (2024-2025)
- **SGMem**: Sentence Graph Memory for long-term conversational agents
- **Nemori**: Self-organizing agent memory inspired by cognitive science
- **RMM**: Reflective Memory Management for personalized dialogue
- **Zep**: Temporal Knowledge Graph Architecture for agent memory

#### Performance Trends
- LongMemEval shows memory systems achieving 90%+ accuracy on memory tasks
- NarrativeQA: State-of-the-art models reach 26.9 F1 score (significant challenge)
- Memory-augmented approaches consistently outperform standard transformers
- Temporal reasoning remains a key challenge across both datasets

### Evaluation Metrics
- Accuracy on long-context tasks
- Memory efficiency
- Computational overhead
- Consistency across context lengths

## Questions to Investigate
- [ ] What memory mechanisms work best for different task types (conversational vs. narrative)?
- [ ] How does performance scale with context length?
- [ ] What are the trade-offs between accuracy and efficiency?
- [ ] How do temporal reasoning capabilities affect long-term memory performance?
- [ ] Can multi-session memory systems transfer to single-document tasks?
- [ ] What role does memory organization (hierarchical vs. flat) play in performance?

## Experimental Design Ideas

### Dataset Comparison Strategy
1. **LongMemEval**: Test conversational memory across sessions
2. **NarrativeQA**: Test narrative understanding and character tracking
3. **Cross-evaluation**: Apply conversational memory systems to narrative tasks and vice versa

### Memory System Categories
- **Session-based**: For multi-turn conversations (LongMemEval focus)
- **Document-based**: For long narrative understanding (NarrativeQA focus)
- **Hybrid**: Systems that handle both scenarios

---
*Last updated: September 29, 2025*