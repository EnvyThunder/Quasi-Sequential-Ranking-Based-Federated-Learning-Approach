# Algorithm 1: Lorenzo (Quasi-Sequential Federated Learning with Jacobi-Seidel Aggregation)

**Input:**
* $T$: Total global iterations
* $E_{boot}$: Epochs for bootstrap (e.g., 1)
* $E_{local}$: Epochs for local training (e.g., 5)
* $K$: Total number of clients
* $\alpha$: Dirichlet parameter for data split
* $B$: Batch size
* $D_{train}^{full}$: Full training dataset (e.g., 50k CIFAR-10)
* $D_{val}^{full}$: Full validation dataset (e.g., 10k split)
* $D_{test}$: Final test dataset
* $\eta$: Learning Rate (e.g., 0.01)

**Helper Functions:**
* `CreateBalancedValLoader(D_{val}, B)`: Applies median over/undersampling to $D_{val}$ and returns a DataLoader.
* `CreateNonIIDSplit(D_{train}, K, \alpha)`: Splits $D_{train}$ among $K$ clients.
* `FilterClients(datasets, min_samples)`: Returns a set of clients $\mathcal{K}$ with $\ge min\_samples$.
* `TrainLocal(W_{in}, L_k, E, \eta)`: Trains model `W_{in}` on client data loader $L_k$ for $E$ epochs using SGD with LR $\eta$. Returns new weights $W_{out}$.
* `Evaluate(W, L_{val}^{bal})$: Returns the F1 score of model `W` on the balanced validation loader.
* `Aggregate(\{W_i\}, \{s_i\})$: Returns the weighted average of models $\{W_i\}$ using scores $\{s_i\}$ as coefficients.

---

**1. Initialization:**
* Initialize `Simple5LayerCNN` model architecture.
* $L_{val}^{bal} \leftarrow \text{CreateBalancedValLoader}(D_{val}^{full}, B)$
* $\{D_k\}_{k=1}^K, \{Labels_k\}_{k=1}^K \leftarrow \text{CreateNonIIDSplit}(D_{train}^{full}, K, \alpha)$
* $\mathcal{K} \leftarrow \text{FilterClients}(\{D_k\}, 2B)$  \Comment{Set of active clients}
* `for` $k \in \mathcal{K}$ `do`:
    * $L_k \leftarrow \text{DataLoader}(D_k[k], B, \text{shuffle=True})$
* $W_g \leftarrow \text{Randomly initialized weights for } \text{Simple5LayerCNN}$

**2. Bootstrap Phase (Round 0):**
* `print("--- Bootstrap Round 0 ---")`
* $H \leftarrow \{\}$  \Comment{Map for historical local weights}
* $S_{boot} \leftarrow \{\}$  \Comment{Map for bootstrap scores}
* `for` $k \in \mathcal{K}$ **in parallel** `do`:
    * $W_k^0 \leftarrow \text{TrainLocal}(W_g, L_k, E_{boot}, \eta)$
    * $H[k] \leftarrow W_k^0$
    * $S_{boot}[k] \leftarrow \text{Evaluate}(W_k^0, L_{val}^{bal})$
* $W_g \leftarrow \text{Aggregate}(\{H[k] \text{ for } k \in \mathcal{K}\}, \{S_{boot}[k] \text{ for } k \in \mathcal{K}\})$
* `print(Evaluate(W_g, L_{val}^{bal}))$` \Comment{Store as "Global Model 0.5" performance}

**3. Main Training Loop:**
* `for` $t = 1 \text{ to } T$ `do`:
    * `print("--- Global Iteration $t$ ---")`
    *
    * **// Step 1: Score and Rank**
    * $S \leftarrow \{\}$  \Comment{Map for current round's scores (from previous history)}
    * `for` $k \in \mathcal{K}$ `do`:
        * $S[k] \leftarrow \text{Evaluate}(H[k], L_{val}^{bal})$  \Comment{$H[k]$ holds weights from round $t-1$}
    * $\mathcal{R} \leftarrow \text{List of client IDs } \mathcal{K} \text{, sorted by } S \text{ descending}$
    *
    * **// Step 2: Sequential Training (Conservative Jacobi-Seidel)**
    * $C \leftarrow \{\}$  \Comment{Map for newly trained models in this round}
    * $s_g \leftarrow \text{Evaluate}(W_g, L_{val}^{bal})$
    * `for` $i = 1 \text{ to } |\mathcal{R}|$ `do`:
        * $k \leftarrow \mathcal{R}[i]$  \Comment{Get client ID at rank $i$}
        * $W_{agg} \leftarrow \{W_g\}$  \Comment{Models to aggregate}
        * $s_{agg} \leftarrow \{s_g\}$  \Comment{Scores for aggregation}
        *
        * **// Add NEW models from peers trained this round**
        * `for` $j = 1 \text{ to } i-1$ `do`:
            * $p \leftarrow \mathcal{R}[j]$  \Comment{Peer at rank $j$ (already trained)}
            * `if` $p \in C$ `then`:
                * $W_{agg}.\text{add}(C[p])$
                * $s_{agg}.\text{add}(S[p])$ \Comment{Use the *updated* score of peer $p$}
        *
        * **// Add OLD models from self and peers not yet trained**
        * `for` $j = i \text{ to } |\mathcal{R}|$ `do`:
            * $p \leftarrow \mathcal{R}[j]$  \Comment{Self (at $j=i$) or peer at rank $j$}
            * `if` $p \in H$ `then`:
                * $W_{agg}.\text{add}(H[p])$
                * $s_{agg}.\text{add}(S[p])$ \Comment{Use the *old* score (from Step 1) of peer $p$}
        *
        * $W_{init}^k \leftarrow \text{Aggregate}(W_{agg}, s_{agg})$
        * $W_k^t \leftarrow \text{TrainLocal}(W_{init}^k, L_k, E_{local}, \eta)$
        * $C[k] \leftarrow W_k^t$
        * $S[k] \leftarrow \text{Evaluate}(W_k^t, L_{val}^{bal})$  \Comment{Update score for use by subsequent peers}
    *
    * **// Step 3: Global Aggregation**
    * $W_g \leftarrow \text{Aggregate}(\{C[k] \text{ for } k \in \mathcal{K}\}, \{S[k] \text{ for } k \in \mathcal{K}\})$
    * $H \leftarrow C$  \Comment{Update history: $H$ now holds models from round $t$}
    * `print(Evaluate(W_g, L_{val}^{bal}))$`

**4. Final Evaluation:**
* `print("--- Final Test Results ---")`
* `print("Global Model:", Evaluate(W_g, \text{DataLoader}(D_{test}, B)))$`
* `for` $k \in \mathcal{K}$ `do`:
    * `print("Client $k$:", Evaluate(H[k], ![Results](https://github.com/user-attachments/assets/e97d0559-0a61-4fe7-8d77-5aaf20db3d78)
\text{DataLoader}(D_{test}, B)))$`

