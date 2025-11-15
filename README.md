\begin{algorithm}[H]
\caption{Lorenzo (Quasi-Sequential FL with Conservative Jacobi-Seidel \& Bootstrap)}
\label{alg:lorenzo_jacobi}
\begin{algorithmic}[1]
\Require
    $T$ (Global iterations), $E_{boot}$ (Bootstrap epochs, e.g., 1), $E_{local}$ (Local epochs, e.g., 5)
\Require
    $K$ (Total clients), $\alpha$ (Dirichlet param), $B$ (Batch size), $D_{\text{train}}^{full}$, $D_{\text{val}}^{full}$, $D_{\text{test}}$, $\eta$ (Learning Rate)
\Require
    $P$ (Early stopping patience, e.g., 10), $\delta$ (Min F1 improvement, e.g., 0.001)

\Statex \textbf{Helper Functions:}
\State \Call{CreateBalancedValLoader}{$D_{val}, B$} $\rightarrow$ Applies median over/undersampling to $D_{val}$ and returns $L_{val}^{bal}$.
\State \Call{CreateNonIIDSplit}{$D_{train}, K, \alpha$} $\rightarrow$ Returns client datasets $\{D_k\}_{k=1}^K$ and labels $\{Labels_k\}_{k=1}^K$.
\State \Call{FilterClients}{$\{D_k\}, min\_samples$} $\rightarrow$ Returns set of active clients $\mathcal{K}$ with $\ge min\_samples$.
\State \Call{TrainLocal}{$W_{in}, L_k, E, \eta$} $\rightarrow$ Trains model $W_{in}$ on $L_k$ for $E$ epochs using SGD (momentum=0, decay=1e-4) with LR $\eta$. Returns new weights $W_{out}$.
\State \Call{Evaluate}{$W, L_{val}^{bal}$} $\rightarrow$ Returns the weighted F1 score $f$ of model $W$ on $L_{val}^{bal}$.
\State \Call{Aggregate}{$\{W_i\}, \{s_i\}$} $\rightarrow$ Returns score-weighted average of models $\{W_i\}$ using coefficients $\{s_i\}$.

\Statex
\State \textbf{Initialization:}
\State $L_{val}^{bal} \leftarrow \Call{CreateBalancedValLoader}{D_{\text{val}}^{full}, B}$
\State $\{D_k\}_{k=1}^K, \{Labels_k\}_{k=1}^K \leftarrow \Call{CreateNonIIDSplit}{D_{\text{train}}^{full}, K, \alpha}$
\State $\mathcal{K} \leftarrow \Call{FilterClients}{\{D_k\}, 2B}$
\State $L \leftarrow \{\}$ \Comment{Map of client DataLoaders}
\ForAll{$k \in \mathcal{K}$}
    \State $L[k] \leftarrow \Call{DataLoader}{D_k[k], B, \text{shuffle=True}}$
\EndFor
\State $W_g \leftarrow \Call{InitializeModel}{\text{Simple5LayerCNN}}$
\State $H \leftarrow \{\}$  \Comment{Map for historical local weights}

\Statex
\State \textbf{Bootstrap Phase (Round 0):}
\State $S_{boot} \leftarrow \{\}$ \Comment{Map for bootstrap scores}
\ForAll{$k \in \mathcal{K}$} \Comment{Can be run in parallel}
    \State $W_k^0 \leftarrow \Call{TrainLocal}{W_g, L[k], E_{boot}, \eta}$
    \State $H[k] \leftarrow W_k^0$
    \State $S_{boot}[k] \leftarrow \Call{Evaluate}{W_k^0, L_{val}^{bal}}$
\EndFor
\State $W_g \leftarrow \Call{Aggregate}{\{H[k]\}_{k \in \mathcal{K}}, \{S_{boot}[k]\}_{k \in \mathcal{K}}}$ \Comment{Create "Global Model 0.5"}
\State $best\_val\_f1 \leftarrow \Call{Evaluate}{W_g, L_{val}^{bal}}$
\State $patience\_counter \leftarrow 0$
\State $W_{best} \leftarrow W_g$ \Comment{Store the bootstrapped model as the first "best" model}

\Statex
\State \textbf{Main Training Loop:}
\For{$t = 0 \to T-1$}
    \State \Comment{\textbf{Step 1: Score and Rank}}
    \State $S \leftarrow \{\}$  \Comment{Map for current round's scores (from previous history)}
    \ForAll{$k \in \mathcal{K}$}
        \State $S[k] \leftarrow \Call{Evaluate}{H[k], L_{val}^{bal}}$  \Comment{$H[k]$ holds weights from round $t$}
    \EndFor
    \State $\mathcal{R} \leftarrow \Call{SortKeysByValue}{S, \text{descending}}$ \Comment{Get ranked list of client IDs}
    
    \State \Comment{\textbf{Step 2: Sequential Training (Conservative Jacobi-Seidel)}}
    \State $C \leftarrow \{\}$  \Comment{Map for newly trained models in this round}
    \State $s_g \leftarrow \Call{Evaluate}{W_g, L_{val}^{bal}}$
    
    \ForAll{$k$ in $\mathcal{R}$} \Comment{Iterate in ranked order}
        \State $W_{agg} \leftarrow \{W_g\}$; $s_{agg} \leftarrow \{s_g\}$ \Comment{Start with global model}
        
        \State \Comment{// Add NEW models (from this round)}
        \ForAll{$p$ in $\mathcal{R}$ before $k$}
            \If{$p \in C$} \Comment{If peer $p$ already trained this round}
                \State $W_{agg}.\text{add}(C[p])$; $s_{agg}.\text{add}(S[p])$ \Comment{Use peer's *updated* score}
            \EndIf
        \EndFor
        
        \State \Comment{// Add OLD models (from previous round)}
        \ForAll{$p$ in $\mathcal{R}$ from $k$ to end} \Comment{Includes self $k$}
            \If{$p \in H$}
                \State $W_{agg}.\text{add}(H[p])$; $s_{agg}.\text{add}(S[p])$ \Comment{Use score from Step 1}
            \EndIf
        \EndFor
        
        \State $W_{init}^k \leftarrow \Call{Aggregate}{W_{agg}, s_{agg}}$
        \State $W_k^t \leftarrow \Call{TrainLocal}{W_{init}^k, L[k], E_{local}, \eta}$
        \State $C[k] \leftarrow W_k^t$
        \State $S[k] \leftarrow \Call{Evaluate}{W_k^t, L_{val}^{bal}}$ \Comment{Update score for next peers in sequence}
    \EndFor
    
    \State \Comment{\textbf{Step 3: Global Aggregation & Early Stopping}}
    \State $W_g \leftarrow \Call{Aggregate}{\{C[k]\}_{k \in \mathcal{K}}, \{S[k]\}_{k \in \mathcal{K}}}$
    \State $H \leftarrow C$  \Comment{Update history for next round}
    
    \State $f1_{curr} \leftarrow \Call{Evaluate}{W_g, L_{val}^{bal}}$
    \State \Call{Print}{"End of Iteration $t+1$: Global F1=$f1_{curr}$"}
    
    \If{$f1_{curr} > best\_val\_f1 + \delta$}
        \State $best\_val\_f1 \leftarrow f1_{curr}$; $patience\_counter \leftarrow 0$; $W_{best} \leftarrow W_g$
        \State \Call{Print}{"New best F1. Patience reset."}
    \Else
        \State $patience\_counter \leftarrow patience\_counter + 1$
        \State \Call{Print}{"No improvement. Patience: $patience\_counter / P$"}
    \EndIf
    
    \If{$patience\_counter \ge P$}
        \State \Call{Print}{"Early stopping triggered."}
        \State \textbf{break} \Comment{Exit main training loop}
    \EndIf
\EndFor

\Statex
\State \textbf{Final Evaluation:}
\State \Call{Print}{"Evaluating best model (F1: $best\_val\_f1$) on Test Set..."}
\State \Call{Evaluate}{W_{best}, \Call{DataLoader}{D_{test}, B}}
\State \Return $W_{best}$
\end{algorithmic}
\end{algorithm}
