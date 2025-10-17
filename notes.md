# Research Notes

## Literature Review

### Survey Memoria LLM
- Main survey paper on LLM memory mechanisms

### Datasets

#### LongMemEval


#### NarrativeQA



### Memory Systems 

#### Mem0
- Importancia de la memoria frente a la ventana de contexto
    - La ventana siempre tendrá un limite, aunque cada vez mayor
    - Hay una perdida de atencion a medida que la cantidad de informacion en la ventana crece
    - No solo es conversacional, sino tambien aprender comportamientos a largo plazo

- Que buscan en un sistema de memoria
    - Que seleccione informacion relevante a almacenar
    - Que consolide la informacion de manera efectiva
    - Que sea capaz de recordar informacion relevante a lo largo del tiempo

##### Como funciona mem0

Para crear una memoria sigue una pipeline de dos fases primero  la **extraccion** de informacion relevante, de la conversacion, y luego la **actualizacion** de la bbdd de memoria usando la informacion extraida y las memorias previas.
    
- Extraccion
    - Usa la ultima interaccion(mensajes nuevos), los ultimos m mensajes y un resumen de la conversacion para *extrear* la informacion relevante de esta interaccion.
    - Esto se le pasa a un LLM que genera una lista de posibles memorias candidatas, de esta nueva interaccion pero que mantengan tematica con la conversacion.

- Actualizacion
    - Para cada candidato de memoria, se hace una busqueda semantica en la bbdd de memoria para recuperar k memorias relacionadas.
   - Usando el candidato y las memorias relacionadas, un LLM decide si:
        1. Ignorar el candidato(**NOOP**)
        2. Actualizar alguna memoria relacionada usando el candidato(**UPDATE**) 
        3. Añadir el candidato como una nueva memoria(**ADD**)
        4. Borrar alguna memoria relacionada(**DELETE**)

##### Como funciona mem0^g

Mem0^g es una version de mem0 que usa una bbdd en grafo para organizar las memorias.

La bbdd esta compuesta por un triple (V, E, L) donde V son las entidades, E las relaciones entre entidades y L las etiquetas de los nodos. <br>
Las entidades tienen tres componentes: un tipo, un significado en embeddings y metadatos. <br>
Las realaciones son triples (entidadOrigen, significadoDeLaRelacion, entidadDestino). <br>


Al igual que en mem0, mem0^g tiene dos fases, extraccion y actualizacion, ambas divididas en dos pasos.
- Extraccion
    - Igual que en mem0, pero el LLM genera una lista de posibles memorias candidatas en forma de triples (entidadOrigen, significadoDeLaRelacion, entidadDestino).

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

---
*Last updated: September 29, 2025*