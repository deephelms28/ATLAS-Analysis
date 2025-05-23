# Game of ATLAS - Strategic Analysis

## Introduction

The Game of ATLAS is a word-based strategic game where players take turns naming countries, with each subsequent country name required to begin with the same letter that the previous country name ended with. For example, if one player says "India," the next player must name a country starting with "a" such as "Australia," followed by a country starting with "a" like "Argentina," and so on. The game continues until a player cannot think of a valid country name that follows this last-letter-to-first-letter rule, at which point that player loses. While seemingly simple, the game involves complex strategic considerations as players must balance maximizing their own future options while simultaneously limiting their opponent's available moves, making it an ideal candidate for graph-theoretic analysis where:

- **Nodes represent countries**
- **Edges exist if the last letter of one country matches the first letter of another**

Using this representation, we can apply graph theory and deep learning techniques to analyze, optimize, and strategize our gameplay. This document outlines four strategic approaches and explores two community detection techniques.

### Competitive Advantage Strategies

1. **Out-Degree Differential Strategy** – Restricting the opponent's viable moves
2. **Entropy Minimization and Forced Predictability** – Forcing opponents into predictable patterns
3. **Centralized Choke-Point Strategy** – Controlling key transition nodes
4. **Local Lookahead Strategy** – Simulating multi-step moves for long-term dominance

### Community Detection Techniques

1. **Infomap** - An information-theoretic approach that models the game as an information flow system
2. **GNN + HDBSCAN** - A deep learning-based method that learns graph embeddings and clusters countries dynamically

Each strategy was simulated in a game, with results visualized through performance metrics and interactive graphs.

## Task 1: Competitive Advantage Strategies

### 1. Out-Degree Differential Strategy

**Goal**: Maximize your own move options while ensuring that the opponent has significantly fewer viable responses.

This strategy moves beyond simply maximizing out-degree by implementing a sophisticated differential analysis. The core principle is to evaluate each potential move not just by how many options it gives you, but by how it affects the relative position between you and your opponent.

**Key Components**:

- **Differential Calculation**: For each potential move, calculate the difference between your best possible out-degree and your opponent's best possible out-degree after they respond
- **Opponent Response Modeling**: Simulate the opponent's optimal response to each of your potential moves
- **Move Advantage Preservation**: Ensure that even when the opponent plays optimally, you maintain a numerical advantage in available moves

**Algorithm Details**:
1. For each valid move from current position:
  - Calculate your out-degree from that position
  - Simulate opponent's best possible response (highest out-degree move)
  - Calculate opponent's out-degree from their best response
  - Compute differential: your_out_degree - opponent_best_out_degree
2. Select the move with the highest positive differential
3. If all differentials are negative, choose the move with the smallest negative differential

**Strategic Implications**: This approach creates sustained pressure on the opponent by consistently limiting their options while preserving your own flexibility. It's particularly effective in mid-to-late game scenarios where move scarcity becomes critical.

![1](https://github.com/user-attachments/assets/a170953d-6787-4650-bbfe-c39f865d18b7)

#### Simulation Results

| Match Type | Player 1 Win Rate | Player 2 Win Rate |
|------------|-------------------|-------------------|
| Both Strategic | 49.10% | 50.90% |
| P1 Strategic vs P2 Random | 63.50% | 36.50% |
| P1 Random vs P2 Strategic | 26.30% | 73.70% |
| Both Random | 48.10% | 51.90% |

### 2. Entropy Minimization and Forced Predictability

**Goal**: Make moves that keep your options open while forcing your opponent into predictable and limited future choices.

This strategy leverages information theory, specifically Shannon entropy, to quantify and minimize the uncertainty in your opponent's future moves. By understanding the probability distribution of your opponent's potential responses, you can make moves that force them into highly predictable patterns.

**Shannon Entropy Implementation**:
$H(X) = -\sum_{x} p(x) \cdot \log_2 p(x)$

Where p(x) represents the probability of the opponent choosing country x as their next move.

**Key Components**:

- **Popularity Weighting**: Countries are weighted by their familiarity and likelihood of being chosen by human players
- **Response Probability Modeling**: Each potential opponent move is assigned a probability based on:
 - Out-degree of the resulting position
 - Country name familiarity/popularity
 - Strategic value of the position
- **Entropy Calculation**: For each of your potential moves, calculate the entropy of your opponent's response distribution
- **Predictability Maximization**: Choose moves that minimize opponent entropy (maximize predictability)

**Algorithm Details**:
1. For each valid move from current position:
  - Identify all valid opponent responses
  - Calculate probability weights for each response based on:
    - Popularity score (pre-computed from common usage)
    - Strategic value (out-degree, centrality measures)
    - Psychological factors (length, pronunciation difficulty)
  - Normalize probabilities to sum to 1.0
  - Calculate Shannon entropy: $H = -\sum_{i} p(i) \cdot \log_2 p(i)$
2. Select the move with the lowest entropy (most predictable opponent response)
3. Among moves with similar entropy, prefer those with higher out-degree for yourself

**Strategic Advantages**: This approach exploits human psychology and the tendency for players to choose familiar, high-value countries. By forcing opponents into predictable choices, you can plan multiple moves ahead with confidence.

![2](https://github.com/user-attachments/assets/2dedf270-589d-4609-bc9c-c82505bd7c31)

#### Simulation Results

| Match Type | Player 1 Win Rate | Player 2 Win Rate |
|------------|-------------------|-------------------|
| Both Entropy | 76.00% | 24.00% |
| P1 Entropy vs P2 Random | 89.00% | 11.00% |
| P1 Random vs P2 Entropy | 21.00% | 79.00% |

### 3. Centralized Choke-Point Strategy

**Goal**: Guide the opponent toward a pre-determined weak position by forcing them through a strategically chosen bottleneck node.

This strategy identifies critical nodes in the graph that serve as mandatory transition points toward opponent disadvantage. Unlike simply choosing high-betweenness nodes, this approach specifically targets nodes that funnel opponents into vulnerable positions.

**Choke-Point Identification Process**:

- **Weak Position Analysis**: Identify positions with low out-degree or high vulnerability
- **Path Frequency Calculation**: For each potential choke point, calculate how many paths from current game states lead through that node to weak positions
- **Composite Scoring**: Combine multiple metrics to rank choke-point effectiveness

**Key Metrics**:

1. **Path Frequency Score**: 
  ```
  PF(node) = (paths_through_node_to_weak_positions) / (total_paths_to_weak_positions)
  ```

2. **Betweenness Centrality**: Measures structural importance in the overall graph

$$BC(\text{node}) = \sum \frac{\sigma_{st}(\text{node})}{\sigma_{st}}$$

Where $\sigma_{st}$ is the number of shortest paths between nodes s and t, and $\sigma_{st}(\text{node})$ is the number of those paths passing through the node

3. **Out-Degree Filter**: Ensures the choke point doesn't trap you
  ```
  OD(node) >= minimum_threshold
  ```

4. **Composite Choke Score**:
  ```
  CS(node) = α * PF(node) + β * BC(node) + γ * OD(node)
  ```
  Where α, β, γ are learned weights optimized through simulation

**Strategic Implementation**:
1. Pre-compute weak positions (out-degree ≤ 2, isolated clusters)
2. Calculate all shortest paths from current position to weak positions
3. Identify nodes that appear in >70% of these paths
4. Rank by composite score and select optimal choke point
5. Plan 2-3 moves ahead to guide opponent toward chosen choke point

**Tactical Considerations**: This strategy requires careful timing - choke points are most effective in mid-game when options are still numerous but patterns are beginning to emerge. Early game applications may be too broad, while late game may offer insufficient path diversity.

### 4. Local Lookahead Strategy

**Goal**: Simulate several turns in advance to find moves that give a long-term advantage while limiting the opponent's flexibility.

This strategy implements a sophisticated multi-move simulation engine that evaluates the downstream consequences of current moves. By looking 2-3 turns ahead, it identifies moves that may appear suboptimal in the short term but provide significant long-term advantages.

**Multi-Metric Evaluation Framework**:

The lookahead algorithm evaluates each potential move across multiple dimensions simultaneously, creating a comprehensive assessment of long-term position strength.

**Core Evaluation Metrics**:

1. **Future Option Differential**:
  ```
  FOD = Σ(t=1 to lookahead_depth) weight(t) * [your_options(t) - opponent_options(t)]
  ```

2. **Spectral Gap Analysis**: Measures graph connectivity to prevent self-trapping
  ```
  SG = λ₂ - λ₁
  ```
  Where λ₁ and λ₂ are the largest and second-largest eigenvalues of the adjacency matrix

3. **Position Stability Score**: Evaluates how robust your position is to opponent counter-strategies
  ```
  PSS = min(your_options) across all opponent responses
  ```

**Simulation Algorithm**:

1. **Move Tree Generation**: For each potential move, generate all possible game continuations up to depth D
2. **Pruning**: Use alpha-beta pruning to eliminate clearly inferior branches
3. **Multi-Metric Scoring**: Evaluate terminal positions using weighted combination of metrics:
  ```
  Score = w₁*FOD + w₂*SG + w₃*PSS + w₄*EndGameProximity
  ```
4. **Minimax with Probability**: Assume opponent plays optimally, but weight moves by likelihood
5. **Best Move Selection**: Choose move with highest expected long-term score

**Advanced Features**:

- **Dynamic Depth Adjustment**: Increase lookahead depth as game progresses and branching factor decreases
- **Position Evaluation Caching**: Store previously computed position evaluations to improve performance
- **Opponent Modeling**: Adapt strategy based on observed opponent patterns
- **Endgame Detection**: Switch to perfect play when game tree becomes fully searchable

**Computational Considerations**: The lookahead strategy is the most computationally intensive, requiring careful optimization. Average branching factor in Atlas is ~8-12, making depth-3 search feasible with proper pruning.

| Candidate | Score | My Spec Gap | Avg Opp Spec | Delta Spec | Avg My Moves | Avg Opp Moves | Delta Moves | H Self | H Opp | Delta Entropy | Composite Score |
|-----------|-------|-------------|--------------|------------|--------------|---------------|-------------|--------|-------|---------------|-----------------|
| iceland | 0.3017 | 0.6737 | 3.2000 | -2.5263 | 10.9440 | 9.4000 | 1.5440 | 3.2679 | 1.9840 | 1.2839 | 0.3017 |
| india | -6.9200 | 0.8768 | 7.5455 | -6.6687 | 11.0975 | 11.1818 | -0.0843 | 3.2914 | 3.4584 | -0.1670 | -6.9200 |
| indonesia | -6.9200 | 0.8768 | 7.5455 | -6.6687 | 11.0975 | 11.1818 | -0.0843 | 3.2914 | 3.4584 | -0.1670 | -6.9200 |
| iran | -3.3591 | 0.8810 | 5.5455 | -4.6644 | 11.6500 | 10.6364 | 1.0137 | 3.4628 | 3.1711 | 0.2917 | -3.3591 |
| iraq | 11.5850 | 3.0000 | 1.0000 | 2.0000 | 12.0000 | 4.0000 | 8.0000 | 1.5850 | 0.0000 | 1.5850 | 11.5850 |
| ireland | 0.3017 | 0.6737 | 3.2000 | -2.5263 | 10.9440 | 9.4000 | 1.5440 | 3.2679 | 1.9840 | 1.2839 | 0.3017 |
| israel | -4.6916 | 1.4893 | 5.5556 | -4.0662 | 11.2727 | 12.5556 | -1.2828 | 3.6215 | 2.9640 | 0.6574 | -4.6916 |
| italy | 1.6842 | 0.8768 | 1.0000 | -0.1232 | 10.6364 | 12.0000 | -1.3636 | 3.1711 | 0.0000 | 3.1711 | 1.6842 |

**Best move from haiti: iraq with composite score 11.5850**

![3](https://github.com/user-attachments/assets/99e9019f-c16f-43d4-927a-c4198fee1478)

#### Simulation Results

| Match Type | Player 1 Win Rate | Player 2 Win Rate |
|------------|-------------------|-------------------|
| P1 Lookahead vs P2 Random | 66.67% | 33.33% |
| P1 Random vs P2 Lookahead | 33.33% | 66.67% |

## Task 2 - Community Detection

### Infomap

#### 1. Modeling Game Flow as Information Flow

**Random Walker Analogy**: Infomap treats the game as a flow of information by simulating a random walker that moves from one country to another following the atlas rule. For instance, if the walker is at "India" (ending with "a"), it can only move to countries that begin with "A" (like "Australia").

**Direction Matters**: Since the atlas graph is directed, Infomap respects the game's rule: moves only happen in the allowed direction (last letter → first letter). This ensures that the flow modeled by the algorithm exactly mimics the possible moves in the game.

The random walker simulation operates by:
1. Starting at a random country
2. Following edges according to atlas rules (last letter matches next first letter)
3. Recording transition frequencies over many iterations
4. Building a flow model that captures the natural movement patterns in the game

**Flow Dynamics**: Countries that are frequently visited in sequence by the random walker are likely to form communities. This captures the natural clustering of countries that work well together in game sequences.

#### 2. The Map Equation and Community Detection

**Minimizing Description Length**: Infomap uses an information-theoretic concept called the map equation. It seeks to partition the graph into communities (clusters of countries) in such a way that the description length of a random walker's journey is minimized. If the walker tends to stay within a certain group of countries, that group is identified as a community.

The **Map Equation** is formulated as:

$$
L(M) = \vec{q}^{\,T} H(Q) + \sum_i p_i^{\circlearrowright} H(P_i)
$$

Where:

- $L(M)$: Description length of a random walk given partition $M$
- $\vec{q}^{\,T} H(Q)$: Entropy of movement **between** communities  
  - $\vec{q}$: Probability vector of exiting modules  
  - $H(Q)$: Shannon entropy over module exits
- $\sum_i p_i^{\circlearrowright} H(P_i)$: Entropy of movement **within** communities  
  - $p_i^{\circlearrowright}$: Probability of staying within community $i$  
  - $H(P_i)$: Shannon entropy of movement within community $i$
- $H(\cdot)$: Represents Shannon entropy


**Two-Level Coding**: The algorithm assigns codes at two levels:
- **Module (Community) Level**: Each community gets a unique code
- **Node Level**: Within each community, each country is given its own code

When the walker moves within a community, only the node-level code is needed. However, moving between communities requires switching to a module-level code. A good community structure is one where most moves are internal, leading to an efficient (compressed) description.

**Optimization Process**: Infomap iteratively refines community assignments to minimize the map equation, using techniques like:
- Simulated annealing for global optimization
- Local search for refinement
- Multi-level aggregation for computational efficiency

#### 3. Interpreting the Computed Communities

**Natural Groupings**: In the atlas graph, communities detected by Infomap often represent clusters of countries that are closely interlinked by the game's rules. For example:

**Letter Patterns**: Countries might group together because many have names that start or end with similar letters. A community could consist mostly of countries that start with "A" or end with "n," reflecting the natural bias in letter transitions.

Common community patterns include:
- **Vowel Starters**: Countries beginning with A, E, I, O, U often cluster together
- **Common Endings**: Countries ending in 'a' (America, China, India) form tight communities
- **Rare Letters**: Countries with Q, X, Z transitions form small, specialized communities

**Choke Points**: Some countries—like Yemen in our example—can act as bridges. Yemen, with incoming edges from countries ending in "y" and outgoing edges to those starting with "n," might form a critical part of a community or even lie at the interface between communities. Such nodes have high betweenness and become strategic choke points.

**Alignment with Human Intuition**: When you look at the computed communities, you might notice that they correspond to groupings that humans would naturally recognize. People often group items based on common features; here, countries with similar starting or ending letters, or those that are strategically central in the flow of the game, naturally fall into the same clusters.

#### 4. Strategic Implications for the Game of Atlas

**Within-Community Moves**: If you know that a group of countries forms a tightly connected community, playing within that community can be a safe strategy. It means you have many follow-up moves because the transitions (edges) are abundant within the community. This strategy minimizes the risk of leaving your opponent with an immediate winning move.

**Exploiting Choke Points**: The Infomap algorithm might reveal that certain countries serve as bridges between communities. Choosing a country that lies on the border (or between) communities could force your opponent into a corner:

- **Limiting Options**: By steering the game through a choke point, you effectively funnel the game into a region where your opponent's valid moves are more restricted
- **Strategic Advantage**: Even if you cannot immediately play the "ideal" country (for example, you may want to say "Hungary" but the current flow does not allow it), you can opt for a move that guides the game toward that region. This long-term planning—based on the community structure—gives you a competitive edge

**Intuition of a "Good" Strategy**: The communities provide insight into the structure of valid moves:

- **Control the Flow**: By understanding which countries are central within a community or connect multiple communities, you can prioritize moves that maximize your control over the game's progression
- **Predictability**: A move within a dense community might give you more predictable follow-up moves, while a move that transitions between communities could be used to disrupt your opponent's expected path

### GAE with GCN and HDBSCAN

#### 1. What is GAE (Graph Autoencoder)?

A Graph Autoencoder (GAE) is a neural network model that learns low-dimensional representations (embeddings) of nodes in a graph while preserving structural relationships. In the context of the Game of Atlas, GAE learns to represent each country as a vector that captures its position and role within the game's strategic landscape.

**Architecture Components**:
- **Encoder**: Compresses node features into a latent space (countries → low-dimensional vectors)
- **Decoder**: Reconstructs edges from embeddings, ensuring learned representations retain graph structure
- **Loss Function**: Minimizes reconstruction error, ensuring embeddings capture key node relationships

The GAE learns by trying to reconstruct the original graph structure from the compressed representations. This forces the encoder to capture the most important structural information about each country's position in the game network.

**Training Process**:
1. **Forward Pass**: Countries are encoded into latent vectors
2. **Reconstruction**: Decoder predicts which countries should be connected
3. **Loss Calculation**: Compare predicted edges with actual atlas rules
4. **Backpropagation**: Update network weights to improve reconstruction
5. **Iteration**: Repeat until convergence

#### 2. Using GCN (Graph Convolutional Networks) as the Encoder

A GCN (Graph Convolutional Network) is a type of neural network designed for graphs, which learns node representations by aggregating information from neighbors. This is particularly well-suited for the Game of Atlas because a country's strategic value depends heavily on its connections to other countries.

**In GAE for the Game of Atlas**:
- **Nodes**: Countries (195 total)
- **Edges**: Valid game transitions (last-letter to first-letter rule)
- **Node Features**: A 52-dimensional one-hot encoding representing:
 - First letter of country name (26 dimensions)
 - Last letter of country name (26 dimensions)

**GCN Layer Operations**:
For each layer l, the node representations are updated as:

$$H^{(l+1)} = \sigma\big(D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H^{(l)} W^{(l)}\big)$$

Where:
- H^(l) are the node features at layer l
- A is the adjacency matrix (atlas connections)
- D is the degree matrix
- W^(l) are learnable weight parameters
- σ is the activation function (ReLU)

**GCN Encoder Layers**:
1. **First Layer (52 → 16)**: Aggregates neighboring features to learn local structure
  - Input: One-hot encoded first/last letters
  - Output: 16-dimensional intermediate representation
  - Learns basic letter transition patterns

2. **Activation (ReLU)**: Introduces non-linearity for complex pattern learning

3. **Second Layer (16 → 4)**: Maps to a latent space (4D vector per country)
  - Input: 16-dimensional intermediate features
  - Output: 4-dimensional final embedding
  - Captures high-level strategic positioning

**Information Aggregation**: Each GCN layer allows countries to "communicate" with their neighbors, building representations that incorporate not just their own properties but also the properties of countries they can transition to and from.

#### 3. HDBSCAN: Clustering the Learned Embeddings

Once the GCN-based encoder learns low-dimensional embeddings for each country, we apply HDBSCAN (Hierarchical Density-Based Clustering) to group similar countries based on their learned representations.

**HDBSCAN Advantages for Game of Atlas**:
- **Handles Noise**: Some countries may not fit cleanly into any community
- **Variable Cluster Sizes**: Natural communities may have very different sizes
- **Hierarchical Structure**: Reveals nested community relationships
- **No Pre-defined K**: Automatically determines optimal number of communities

**Algorithm Steps**:

1. **Core Distance Calculation**: For each country embedding, compute the distance to its k-th nearest neighbor
  ```
  core_distance(p) = distance(p, kth_nearest_neighbor(p))
  ```

2. **Mutual Reachability Distance**: Define distance between countries considering local density
  ```
  d_mreach(a,b) = max(core_distance(a), core_distance(b), distance(a,b))
  ```

3. **Minimum Spanning Tree**: Build MST using mutual reachability distances

4. **Cluster Hierarchy**: Create dendrogram by iteratively removing edges

5. **Stability-Based Selection**: Choose cluster structure with maximum stability
  ```
  stability(C) = Σ(λ_death - λ_birth) for all points in cluster C
  ```

6. **Flat Clustering**: Extract final communities from hierarchical structure

**Parameter Tuning**:
- **min_cluster_size**: Minimum countries per community (typically 3-5)
- **min_samples**: Core point threshold for noise detection
- **cluster_selection_epsilon**: Distance threshold for cluster merging

**Interpretation of Results**: The final communities represent countries that have similar strategic roles in the game. Countries in the same cluster likely:
- Have similar connectivity patterns
- Occupy similar positions in game flow
- Present similar strategic opportunities/challenges
- Can often be used interchangeably in tactical situations

#### 4. Combined GAE+HDBSCAN Strategic Insights

The combination of GAE and HDBSCAN provides a data-driven approach to understanding game structure that complements the theoretical analysis from Infomap.

**Strategic Applications**:

1. **Substitution Strategies**: Countries in the same cluster can often substitute for each other in planned sequences
2. **Community Transitions**: Understanding which clusters connect helps plan multi-move strategies
3. **Vulnerability Analysis**: Isolated countries (noise points) represent high-risk/high-reward moves
4. **Dynamic Adaptation**: Embeddings can be retrained as gameplay data accumulates, improving strategy over time

## Conclusion

This analysis demonstrates that sophisticated graph-theoretic and machine learning approaches can significantly improve performance in the Game of ATLAS. The entropy minimization strategy showed the highest win rates (up to 89% against random play), while community detection methods like Infomap and GAE+HDBSCAN reveal structural insights that enable long-term strategic planning.

**Key Findings**:
- Entropy-based strategies outperform simple out-degree maximization
- Community structure reveals natural strategic groupings
- Lookahead strategies provide significant advantages with manageable computational cost
- Machine learning approaches (GAE+HDBSCAN) complement traditional graph analysis

The combination of these approaches provides a comprehensive framework for understanding and mastering the Game of ATLAS through computational analysis.
