We will here discuss the reasoning behind the two scaling factor used in the \Normalizing Flows\src\flows.py python script, at the end of the neural networks used as s and m functions in affine-coupling, conditional normalizing flows

The total flow $f$ is a composition of invertible functions $f = f_N(f_{N-1}(...(f_1)))$

Here $y_0$ is the input and $y_n$ is $f_n(y_{n-1})$ such that $y_N = f(y_0)$ is the output of the flow.

$y_n = concat(y_{n,1},y_{n,2})$ is given by the formula : $$ y_{n,1} = y_{n-1,1} . exp(s_1(y_{n-1,2})) + m_1(y_{n-1,2});\quad  y_{n,2} = y_{n-1,2} . exp(s_1(y_{n,1})) + m_1(y_{n,1})$$

Let $||||$ be the infinite-norm over $R^d$ where d is the dimension of the input of the normalizing flow.

Let $s$ and $m$ be the maximum coordinate of both of the $s_i$ and $abs(m_i)$ functions (we assume such maximums exist).

$$ ||y_n|| <= e^s ||y_{n-1}|| + m  $$

This is an Arithmetic-geometric sequence, thus $||y_N|| <= (e^s)^N ||y_{0}|| + ((e^s)^N - 1)/(e^s - 1) * m $

As the flow is built to transition between two normalized probability density functions, we can assume that the flow does not require large affine constants $m$ to function properly as both distributions are centered and few "translations" should be needed.

Thus we will normalize both  $m_1$ and $m_2$ through the use of a sigmoid, such that $m = 1$.

$||y_N|| <= (e^s)^N ||y_{0}|| + ((e^s)^N - 1)/(e^s - 1)$

We want to bound $||y_N||$ by a factor M i.e we want: $||y_N|| <= M ||y_0||$. As y_0 is a normal gaussian, we know that $||y_0||$ is $O(1)$ so we will consider ||y_{0}|| = 1 to perform rough approximate computations.
Let's define $a = e^s$

$$||y_N|| <= a^N ||y_{0}|| + (a^N - 1)/(a - 1) <= M ||y_0||$$

So we want $a^N + (a^N - 1)/(a - 1) <= M; \quad a^N(a-1) + a^N-1 <= M(a-1); \quad a^{n+1} - aM <= 1 - M $

This is notably true for a = 1. Furthermore the derivative of the second member is negative for a = 1 and becomes null when $ a = (M/(N+1))^{1/N} > 1 $ as M comes from a limitation of the float max value and is therefore much bigger than the flow length N

thus we want $ e^s = (M/(N+1))^{1/N}$ i.e $ s = 1/N * ln (M/(N+1)) $

Thus we will normalize both  $s_1$ and $s_2$ through the use of a sigmoid, then scale it down by a factor $1/N * ln (M/(N+1))$