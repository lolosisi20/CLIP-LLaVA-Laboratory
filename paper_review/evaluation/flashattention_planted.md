# FlashAttention: Fast and Memory-Efficient Exact Attention

# Abstract
FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness
Tri Dao†, Daniel Y. Fu†, Stefano Ermon†, Atri Rudra‡, and Christopher Ré†
†Department of Computer Science, Stanford University
‡Department of Computer Science and Engineering, University at Buffalo, SUNY
{trid,danfu}@cs.stanford.edu, ermon@stanford.edu, atri@buffalo.edu,
chrismre@cs.stanford.edu
June 24, 2022
Abstract
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity
of self-attention are quadratic in sequence length. Approximate attention methods have attempted
to address this problem by trading offmodel quality to reduce the compute complexity, but often do
not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention,
an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes
between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity
of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is
optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding
an approximate attention algorithm that is faster than any existing approximate attention method.
FlashAttention trains Transformers faster than existing baselines: 85% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 30× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K). FlashAttention
and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models
(0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new
capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge
(seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

# 1 Introduction
Transformer models [82] have emerged as the most widely used architecture in applications such as natural
language processing and image classification. Transformers has grown larger [5] and deeper [83], but
equipping them with longer context remains difficult [80], since the self-attention module at their heart
has time and memory complexity quadratic in sequence length. An important question is whether making
attention faster and more memory-efficient can help Transformer models address their runtime and memory
challenges for long sequences.
Many approximate attention methods have aimed to reduce the compute and memory requirements of
attention. These methods range from sparse-approximation [51, 74] to low-rank approximation [12, 50, 84],
and their combinations [3, 9, 92]. Although these methods reduce the compute requirements to linear or
near-linear in sequence length, many of them do not display wall-clock speedup against standard attention
and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not
correlate with wall-clock speed) and tend to ignore overheads from memory access (IO).
In this paper, we argue that a missing principle is making attention algorithms IO-aware [1]—that is,
carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU
on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [45], Figure 1 left). On modern
1
arXiv:2205.14135v2 [cs.LG] 23 Jun 2022

FlashAttention
Memory Hierarchy with
Bandwidth & Memory Size
Attention on GPT-2
FlashAttention
PyTorch
Time (ms)
Matmul
Mask
Softmax
Dropout
Matmul
Fused
Kernel
Q: N x d
V: N X d
KT: d x N
QKT: N x N
sm(QKT)V: N x d
Outer Loop
Copy Block to SRAM
Copy
Outer Loop
Copy
Inner Loop
Compute Block
on SRAM
Output to HBM
Inner Loop
Inner Loop
Outer Loop
GPU
SRAM
GPU
HBM
Main Memory
(CPU DRAM)
SRAM: 19 TB/s (20 MB)
HBM: 1.5 TB/s (40 GB)
DRAM: 12.8 GB/s
 (>1 TB)
0
5
10
15
Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large N× Nattention matrix
(dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through
blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention
loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention
computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2.
FlashAttention does not read and write the large N× Nattention matrix to HBM, resulting in an 7.6×
speedup on the attention computation.
GPUs, compute speed has out-paced memory speed [61, 62, 63], and most operations in Transformers are bottlenecked by memory acceses [43]. IO-aware algorithms have been critical for similar memory-bound
operations, when reading and writing data can account for a large portion of the runtime—such as database
joins [71], image processing [70], numerical linear algebra [4], and more [40, 85]. However, common Python
interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory
access.
We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer
memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM.
This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large
intermediate attention matrix for the backward pass. We apply two well-established techniques to address
these challenges. (i) We restructure the attention computation to split the input into blocks and make several
passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling). (ii) We
store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the
backward pass, which is faster than the standard approach of reading the intermediate attention matrix from
HBM. We implement FlashAttention in CUDA to achieve fine-grained control over memory access and
fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation,
our algorithm both runs faster (up to 7.6x on GPT-2 [67], Figure 1 right) and uses less memory—linear
in sequence length—than standard attention, thanks to the massively reduced amount of HBM access.
We analyze the IO complexity [1] of FlashAttention, proving that it requires O(N2d2M−1) HBM
accesses where dis the head dimension and Mis the size of SRAM, as compared to Ω(Nd+ N2) of standard
attention. For typical values of dand M, FlashAttention requires many times fewer HBM accesses
compared to standard attention (up to 9× fewer, as shown in Fig. 2). Moreover, we provide a lower bound,
showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over
all SRAM sizes.
We also show that FlashAttention can serve as a useful primitive for realizing the potential of
approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of
concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4× faster than
even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention
has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss
further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix
2

multiply) in Section 5. We open-source FlashAttention to make it more easier to build on this primitive.1
We empirically validate that FlashAttention speeds up model training and improves model quality by
modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and
block-sparse FlashAttention compared to prior attention implementations.
• Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We
train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [58], GPT2
(seq. length 1K) 3× faster than baseline implementations from HuggingFace [87] and Megatron-LM [77],
and long-range arena (seq. length 1K-4K) 2.4× faster than baselines.
• Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves
their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and
6.4 points of lift from modeling longer sequences on long-document classification [13]. FlashAttention
enables the first Transformer that can achieve better-than-chance performance on the Path-X [80] challenge,
solely from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer
to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance
performance on Path-256.
• Benchmarking Attention. FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512,
FlashAttention is both faster and more memory-efficient than any existing attention method, whereas
for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become
faster. On the other hand, block-sparse FlashAttention is faster than all existing approximate attention
methods that we know of.

# 2 Background
We provide some background on the performance characteristics of common deep learning operations on
modern hardware (GPUs). We also describe the standard implementation of attention.

# 2.1 Hardware Performance
We focus here on GPUs. Performance on other hardware accelerators are similar [46, 48].
GPU Memory Hierarchy. The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of
memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 4-8GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM
per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45]. The on-chip SRAM is an order of magnitude slower than HBM but many orders of magnitude smaller in size. As compute
has gotten faster relative to memory speed [61, 62, 63], operations are increasingly bottlenecked by memory
(HBM) accesses. Thus exploiting fast SRAM becomes more important.
Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel).
Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.
Performance characteristics. Depending on the balance of computation and memory accesses, operations can be classified as either compute-bound or memory-bound. This is commonly measured by the
arithmetic intensity [85], which is the number of arithmetic operations per second of compute time.
1. Compute-bound: the time taken by the operation is determined by how many arithmetic operations there
are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner
dimension, and convolution with large number of channels.
2. Memory-bound: the time taken by the operation is determined by the number of memory accesses, while
time spent in computation is much smaller. Examples include most other operations: elementwise (e.g.,
activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).
Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if
there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of
multiple times for each operation. Compilers can automatically fuse many elementwise operations [53, 65, 75].
1FlashAttention code is available at https://github.com/HazyResearch/flash-attention
3

However, in the context of model training, the intermediate values still need to be written to HBM to save
for the backward pass, reducing the effectiveness of naive kernel fusion.

# 2.2 Standard Attention Implementation
Given input sequences Q, K, V ∈RN×dwhere Nis the sequence length and dis the head dimension, we want
to compute the attention output O ∈RN×d:
S = QK⊤∈RN×N,
P = softmax(S) ∈RN×N,
O = PV ∈RN×d,
where softmax is applied row-wise.
Standard attention implementations materialize the matrices S and P to HBM, which takes O(N2) memory.
Often N≫d(e.g., for GPT2, N= 1024 and d= 64). We describe the standard attention implementation
in Algorithm 0. As some or most of the operations are memory-bound (e.g., softmax), the large number of
memory accesses translates to slow wall-clock time.
This problem is exacerbated by other elementwise operations applied to the attention matrix, such as
masking applied to S or dropout applied to P. As a result, there have been many attempts to fuse several
elementwise operations, such as fusing masking with softmax [77].
In Section 3.2, we will show that the standard attention implementation performs HBM accesses quadratic
in the sequence length N. We also compare the number of FLOPs and number of HBM accesses of standard
attention and of our method (FlashAttention).
Algorithm 0 Standard Attention Implementation
Require: Matrices Q, K, V ∈RN×din HBM.
1: Load Q, K by blocks from HBM, compute S = QK⊤, write S to HBM.
2: Read S from HBM, compute P = softmax(S), write P to HBM.
3: Load P and V by blocks from HBM, compute O = PV, write O to HBM.
4: Return O.

# 3 FlashAttention: Algorithm, Analysis, and Extensions
We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate
matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in
wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses
compared to standard attention. We further show that FlashAttention can serve as a useful primitive by
extending it to handle block-sparse attention.
We focus here on the forward pass for ease of exposition; Appendix B contains details for the backward.

# 3.1 An Efficient Attention Algorithm With Tiling and Recomputation
Given the inputs Q, K, V ∈RN×din HBM, we aim to compute the attention output O ∈RN×dand write it to
HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in N).
We apply two established techniques (tiling, recomputation) to overcome the technical challenge of
computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1. The main idea
is that we split the inputs Q, K, V into blocks, load them from slow HBM to fast SRAM, then compute the
attention output with respect to those blocks. By scaling the output of each block by the right normalization
factor before adding them up, we get the correct result at the end.
Tiling. We compute attention by blocks. Softmax couples columns of K, so we decompose the large
softmax with scaling [51, 60, 66].
For numerical stability, the softmax of vector x∈RBis computed as:
m(x) := max
i
xi,
f(x) :=

ex1−m(x)
. . .
exB−m(x)
,
l(x) :=
∑︁
i
f(x)i,
softmax(x) := f(x)
l(x) .
4

For vectors x(1), x(2) ∈RB, we can decompose the softmax of the concatenated x=

x(1) x(2)
∈R2Bas:
m(x) = m(

x(1) x(2)
) = max(m(x(1)), m(x(2))),
f(x) =
h
em(x(1))−m(x) f(x(1))
em(x(2))−m(x) f(x(2))
i
,
l(x) = l(

x(1) x(2)
) = em(x(1))−m(x)l(x(1)) + em(x(2))−m(x)l(x(2)),
softmax(x) = f(x)
l(x) .
Therefore if we keep track of some extra statistics (m(x), l(x)), we can compute softmax one block at a time.2
We thus split the inputs Q, K, V into blocks (Algorithm 1 line 3), compute the softmax values along with
extra statistics (Algorithm 1 line 10), and combine the results (Algorithm 1 line 12).
Recomputation. One of our goals is to not store O(N2) intermediate values for the backward pass. The
backward pass typically requires the matrices S, P ∈RN×Nto compute the gradients with respect to Q, K, V.
However, by storing the output O and the softmax normalization statistics (m, l), we can recompute the
attention matrix S and P easily in the backward pass from blocks of Q, K, V in SRAM. This can be seen as a
form of selective gradient checkpointing [10, 34]. While gradient checkpointing has been suggested to reduce
the maximum amount of memory required [66], all implementations (that we know off) have to trade speed
for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to
reduced HBM accesses (Fig. 2). The full backward pass description is in Appendix B.
Implementation details:
Kernel fusion.
Tiling enables us to implement our algorithm in one
CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax,
optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout
in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM.
Algorithm 1 FlashAttention
Require: Matrices Q, K, V ∈RN×din HBM, on-chip SRAM of size M.
1: Set block sizes Bc=
 M
4d

, Br= min   M
4d

, d.
2: Initialize O = (0)N×d∈RN×d, l= (0)N∈RN, m= (−∞)N∈RNin HBM.
3: Divide Q into Tr=
l
N
Br
m
blocks Q1, . . . , QTrof size Br× deach, and divide K, V in to Tc=
l
N
Bc
m
blocks
K1, . . . , KTcand V1, . . . , VTc, of size Bc× deach.
4: Divide O into Trblocks Oi, . . . , OTrof size Br× deach, divide linto Trblocks li, . . . , lTrof size Breach,
divide minto Trblocks m1, . . . , mTrof size Breach.
5: for 1 ≤j≤Tcdo
6:
Load K j, Vjfrom HBM to on-chip SRAM.
7:
for 1 ≤i≤Trdo
8:
Load Qi, Oi, li, mifrom HBM to on-chip SRAM.
9:
On chip, compute Sij= QiKT
j∈RBr×Bc.
10:
On chip, compute ̃mij= rowmax(Sij) ∈RBr, ̃Pij= exp(Sij− ̃mij) ∈RBr×Bc(pointwise), ̃lij=
rowsum( ̃Pij) ∈RBr.
11:
On chip, compute mnew
i
= max(mi, ̃mij) ∈RBr, lnew
i
= emi−mnew
i
li+ e ̃mij−mnew
i
 ̃lij∈RBr.
12:
Write Oi←diag(lnew
i
)−1(diag(li)emi−mnew
i
Oi+ e ̃mij−mnew
i
 ̃PijVj) to HBM.
13:
Write li←lnew
i
, mi←mnew
i
to HBM.
14:
end for
15: end for
16: Return O.
We show FlashAttention’s correctness, runtime, and memory requirement (proof in Appendix C).
Theorem 1. Algorithm 1 returns O = softmax(QK⊤)V with O(N2d) FLOPs and requires O(N) additional
memory beyond inputs and output.

# 3.2 Analysis: IO Complexity of FlashAttention
We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared
to standard attention. We also provide a lower bound, proving that no exact attention algorithm can
2This style of aggregation is called algebraic aggregation [33].
5

Attention
Standard
FlashAttention
GFLOPs
66.6
75.2
HBM R/W (GB)
40.3
4.4
Runtime (ms)
41.7
7.3
Sparsity Speedup
% Non-Zero Blocks
20
60
50
100
150
Fwd + Bwd (ms)
Effect of Block Size
Block Size
64 128
256
512
Fwd Runtime (ms)
6
2
HBM Accesses (GB)
Dense
FlashAttention
Block-Sparse
FlashAttention
2
4
6
Runtime
HBM
Accesses
Figure 2:
Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium
(seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor affecting
runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on
A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of
block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity.
asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C.
Theorem 2. Let Nbe the sequence length, dbe the head dimension, and Mbe size of SRAM with d≤M≤Nd.
Standard attention (Algorithm 0) requires Θ(Nd+ N2) HBM accesses, while FlashAttention (Algorithm 1)
requires Θ(N2d2M−1) HBM accesses.
For typical values of d(64-128) and M(around 100KB), d2 is many times smaller than M, and thus
FlashAttention requires many times fewer HBM accesses than standard implementation. This leads to
both faster execution and lower memory footprint, which we validate in Section 4.3.
The main idea of the proof is that given the SRAM size of M, we can load blocks of K, V of size Θ(M) each
(Algorithm 1 line 6). For each block of K and V, we iterate over all blocks of Q (Algorithm 1 line 8) to compute
the intermediate values, resulting in Θ(NdM−1) passes over Q. Each pass loads Θ(Nd) elements, which
amounts to Θ(N2d2M−1) HBM accesses. We similarly prove that the backward pass of standard attention
requires Θ(Nd+ N2) HBM accesses while the backward pass of FlashAttention requires Θ(N2d2M−1)
HBM accesses (Appendix B).
We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all
values of M(the SRAM size) when computing exact attention.
Proposition 3. Let Nbe the sequence length, dbe the head dimension, and Mbe size of SRAM with
d≤M≤Nd. There does not exist an algorithm to compute exact attention with o(N2d2M−1) HBM accesses
for all Min the range [d, Nd].
The proof relies on the fact that for M= Θ(Nd) any algorithm must perform Ω(N2d2M−1) = Ω(Nd)
HBM accesses. This type of lower bound over a subrange of Mis common in the streaming algorithms
literature [88]. We leave proving parameterized complexity [27] lower bounds in terms of Mas exciting future
work.
We validate that the number of HBM accesses is the main determining factor of attention run-time.
In Fig. 2 (left), we see that even though FlashAttention has higher FLOP count compared to standard
attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much
faster runtime. In Fig. 2 (middle), we vary the block size Bcof FlashAttention, which results in different
amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number
of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough
block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations).
Moreover, larger block size will not fit into the small SRAM size.

# 3.3 Extension: Block-Sparse FlashAttention
We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention, whose
IO complexity is smaller than FlashAttention by a factor proportional to the sparsity.
Given inputs Q, K, V ∈RN×dand a mask matrix ̃M ∈{0, 1}N×N, we want to compute:
S = QK⊤∈RN×N,
P = softmax(S ⊙1 ̃M) ∈RN×N,
O = PV ∈RN×d,
where (S ⊙1 ̃M)kl= Sklif ̃Mkl= 1 and −∞if Mkl= 0. We require ̃M to have block form: for some block sizes
Br, Bc, for all k, l, ̃Mk,l= Mijwith i= ⌊k/Br⌋, j= ⌊l/Bc⌋for some M ∈{0, 1}N/Br×N/Bc.
6

Given a predefined block sparsity mask M ∈{0, 1}N/Br×N/Bcwe can easily adapt Algorithm 1 to only
compute the nonzero blocks of the attention matrix. The algorithm is identical to Algorithm 1, except we
skip zero blocks. We reproduce the algorithm description in Algorithm 5 in Appendix B.
We also analyze the IO complexity of block-sparse FlashAttention.
Proposition 4. Let Nbe the sequence length, dbe the head dimension, and Mbe size of SRAM with
d≤M≤Nd. Block-sparse FlashAttention (Algorithm 5) requires Θ(Nd+ N2d2M−1s) HBM accesses
where sis the fraction of nonzero blocks in the block-sparsity mask.
We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the
IO complexity. For large sequence lengths N, sis often set to N−1/2 [11] or N−1 log N[3, 17, 92], resulting
in Θ(N
√
N) or Θ(Nlog N) IO complexity. For downstream experiments, we use the fixed butterfly sparsity
pattern [17], which has been shown to be able to approximate arbitrary sparsity [16].
In Fig. 2 (right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention
improves proportionally. On the LRA benchmark, block-sparse FlashAttention achieves 2.8× speedup,
while performing on par with standard attention (Section 4).

# 4 Experiments
We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims
about training time and model accuracy, and report attention runtime and memory benchmarks.
• Training Speed. FlashAttention outperforms the MLPerf 1.1 [58] speed record for BERT by 15%, and
speeds up GPT-2 up to 3× over HuggingFace [87] and 1.8× over Megatron [77] over standard Transformers.
FlashAttention speeds up the long-range arena (LRA) benchmark 2.4×.
• Quality. FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length
1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two longdocument classification tasks. Finally, FlashAttention yields the first Transformer that can achieve
better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse
FlashAttention yields the first sequence model that we know of that can achieve better-than-random
performance on Path-256 (sequence length 64K).
• Benchmarking Attention. We measure the runtime and memory performance of FlashAttention
and block-sparse FlashAttention based on sequence length. We confirm that the memory footprint
of FlashAttention scales linearly with seq. length and is up to 3× faster than standard attention for
common seq. lengths (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly
in seq. length and is faster than all existing approximate attention baselines.
Additional experiment details are in Appendix E.

# 4.1 Faster Models with FlashAttention
BERT.
FlashAttention yields the fastest single-node BERT training speed that we know of. We train a
BERT-large [22] model with FlashAttention on Wikipedia. Table 1 compares our training time to the
implementation from Nvidia that set the training speed record for MLPerf 1.1 [58]. Our implementation is
15% faster.
Table 1: Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to
reach the target accuracy of 72.0% on masked language modeling. Averaged over 10 runs on 8×A100 GPUs.
BERT Implementation
Training time (minutes)
Nvidia MLPerf 1.1 [58]
20.0 ± 1.5
FlashAttention (ours)
17.4 ± 1.4
GPT-2.
FlashAttention yields faster training times for GPT-2 [67] on the large OpenWebtext dataset [32]
than the widely used HuggingFace [87] and Megatron-LM [77] implementations. Table 2 shows up to 3× endto-end speedup compared to Huggingface and 1.7× speedup compared to Megatron-LM. FlashAttention
7

achieves the same perplexity as the other two implementations, as we do not change the model definition.
Appendix E includes plots of the validation perplexity throughout training, confirming that FlashAttention
is as numerically stable as the baselines and produces the same training / validation curves.
Table 2: GPT-2 small and medium using FlashAttention achieve up to 3× speed up compared to Huggingface
implementation and up to 1.7× compared to Megatron-LM. Training time reported on 8×A100s GPUs.
Model implementations
OpenWebText (ppl)
Training time (speedup)
GPT-2 small - Huggingface [87]
18.2
9.5 days (1.0×)
GPT-2 small - Megatron-LM [77]
18.2
4.7 days (2.0×)
GPT-2 small - FlashAttention
18.2
2.7 days (3.5×)
GPT-2 medium - Huggingface [87]
14.2
21.0 days (1.0×)
GPT-2 medium - Megatron-LM [77]
14.3
11.5 days (1.8×)
GPT-2 medium - FlashAttention
14.3
6.9 days (3.0×)
Long-range Arena.
We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA [80]) benchmark. We measure accuracy, throughput, and training
time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the
implementation and experimental setting in Tay et al. [80]and Xiong et al. [90].3 Table 3 shows that FlashAttention achieves up 2.4× speed-up compared to standard attention. Block-sparse FlashAttention is
faster than all of the approximate attention methods that we have tested.
Table 3: The performance of standard attention, FlashAttention, block-sparse FlashAttention, and approximate
attention baselines on the Long-Range-Arena benchmarks.
Models
ListOps
Text
Retrieval
Image
Pathfinder
Avg
Speedup
Transformer
36.0
63.6
81.6
42.3
72.7
59.3
-
FlashAttention
37.6
63.9
81.4
43.5
72.7
59.8
2.4×
Block-sparse FlashAttention
37.0
63.0
81.3
43.6
73.3
59.6
2.8×
Linformer [84]
35.6
55.9
77.7
37.8
67.6
54.9
2.5×
Linear Attention [50]
38.8
63.2
80.7
42.6
72.5
59.6
2.3×
Performer [12]
36.8
63.6
82.2
42.1
69.9
58.9
1.8×
Local Attention [80]
36.1
60.2
76.7
40.6
66.6
56.0
1.7×
Reformer [51]
36.5
63.8
78.5
39.6
69.4
57.6
1.3×
Smyrf [19]
36.1
64.1
79.0
39.6
70.5
57.9
1.7×

# 4.2 Better Models with Longer Sequences
Language Modeling with Long Context.
The runtime and memory-efficiency of FlashAttention
allow us to increase the context length of GPT-2 by 4× while still running faster than the optimized
implementation from Megatron-LM. Table 4 shows that that GPT-2 with FlashAttention and context
length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better
perplexity.
Table 4: GPT-2 small with FlashAttention, with 4× larger context length compared to Megatron-LM, is still 30%
faster while achieving 0.7 better perplexity. Training time on 8×A100 GPUs is reported.
Model implementations
Context length
OpenWebText (ppl)
Training time (speedup)
GPT-2 small - Megatron-LM
1k
18.2
4.7 days (1.0×)
GPT-2 small - FlashAttention
1k
18.2
2.7 days (1.7×)
GPT-2 small - FlashAttention
2k
17.6
3.0 days (1.6×)
GPT-2 small - FlashAttention
4k
17.5
3.6 days (1.3×)
Long Document Classification.
Training Transformers with longer sequences with FlashAttention
improves performance on the MIMIC-III [47] and ECtHR [6, 7] datasets. MIMIC-III contains intensive care
unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the
3LRA accuracy results are known to be highly dependent on the tuning procedure [90]. Our reproduced baselines perform
better than as reported in the original comparison [80].
8

Attention Memory Usage
Sequence Length
Attention Runtime (Fwd Pass + Bwd Pass)
Sequence Length
Runtime (ms)
Memory Footprint (GB)
256
8K
16K
32K
64K
128
256
512
1024
2048
4096
101
102
10
20
FlashAttention
Block-Sparse FlashAttention
PyTorch Attention
Megatron Attention
Linformer Attention
OpenAI Sparse Attention
8192
100
Crossover Points
20x
2x
Figure 3: Left: runtime of forward pass + backward pass. Right: attention memory usage.
European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights
that were allegedly violaged. Both of these datasets contain very long text documents; the average number of
tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and
longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence
length of a pretrained RoBERTa model [56] (we repeat the positional embeddings, as in Beltagy et al. [3]).
Table 5 shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length
8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution
shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift
in the document length, whereas ECtHR contains general language.
Table 5: Long Document performance (micro F1) at different sequence lengths using
FlashAttention.
512
1024
2048
4096
8192
16384
MIMIC-III [47]
52.8
50.7
51.7
54.6
56.4
57.1
ECtHR [6]
72.2
74.3
77.1
78.6
80.7
79.2
Table 6:
We report the first Transformer
model that can achieve non-random performance on Path-X and Path-256.
Model
Path-X
Path-256
Transformer


Linformer [84]


Linear Attention [50]


Performer [12]


Local Attention [80]


Reformer [51]


SMYRF [19]


FlashAttention
61.4

Block-sparse FlashAttention
56.0
63.1
Path-X and Path-256.
The Path-X and Path-256 benchmarks are challenging tasks from the long-range
arena benchmark designed to test long context. The task is to classify whether two points in a black and
white 128×128 (or 256×256) image have a path connecting them, and the images are fed to the transformer
one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved
random performance [80]. There has been a search for alternative architectures that can model such long
context [37]. We present here the first result of Transformer models being able to solve Path-X and Path-256
(Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating
the positional embeddings. FlashAttention achieves 61.4 accuracy on Path-X. Additionally, block-sparse
FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy4 on
Path-256.

# 4.3 Benchmarking Attention
GPU Memory Hierarchy. The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of
memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU
has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM
per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45]. The on-chip
SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute
has gotten faster relative to memory speed [61, 62, 63], operations are increasingly bottlenecked by memory
(HBM) accesses. Thus exploiting fast SRAM becomes more important.

We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse
FlashAttention against various attention baselines on one A100 GPU with 40 GB HBM, with dropout and
a padding mask. We compare against reference implementations for exact attention, approximate attention,
and sparse attention. We report a subset of baselines in the main body; Appendix E contains more baselines
and full details.
4Path-256 requires longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy.
9

Runtime.
Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse
attention (exact numbers in Appendix E). Runtime grows quadratically with sequence length, but FlashAttention runs significantly faster than exact attention baselines, up to 3× faster than the PyTorch
implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than approximate and sparse attention for short
sequences due to fewer memory accesses. The approximate attention runtimes begin to cross over with
FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention
is faster than all implementations of exact, sparse, and approximate attention that we know of, across all
sequence lengths.
Memory Footprint.
Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse
FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention
and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence
length. FlashAttention is up to 20× more memory efficient than exact attention baselines, and is more
memory-efficient than the approximate attention baselines. All other algorithms except for Linformer run
out of memory on an A100 GPU before 64K, and FlashAttention is still 2× more efficient than Linformer.

# 5 Limitations and Future Directions
We discusses limitation of our approach and future directions. Related work is given in Appendix A.
Compiling to CUDA. Our current approach to building IO-aware implementations of attention requires
writing a new CUDA kernel for each new attention implementation. This requires writing the attention
algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort.
Implementations may also not be transferrable across GPU architectures. These limitations suggest the
need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and
compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [70].
IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention.
Attention is the most memory-intensive computation in Transformers, but every layer in a deep network
touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss
these potential extensions in Appendix D.
Multi-GPU IO-Aware Methods. Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable
across multiple GPUs [72]. Using multiple GPUs adds an additional layer to IO analysis—accounting for
data transfer between GPUs. We hope our work inspires future work in this direction.