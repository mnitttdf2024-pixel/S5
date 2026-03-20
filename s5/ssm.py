from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_sequence,
                       self.conj_sym,
                       self.bidirectional)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def apply_ssm_adaptive(Lambda_bar, B_bar, C_tilde, W_adapt, b_adapt,
                       input_sequence, conj_sym):
    """Compute the LxH output of a state-regulated SSM using a sequential scan.

    At each step t the forgetting strength is adjusted based on three diagnostics
    computed from the two most recent hidden states:

        r_t  = ||h_{t-1} - h_{t-2}|| / (||h_{t-2}|| + eps)   relative change
        q_t  = ||h_{t-1}||^2 / (||h_{t-2}||^2 + eps)          energy ratio
        c_t  = Re(h_{t-1}^H h_{t-2}) / (||h_{t-1}||*||h_{t-2}|| + eps)  cosine sim

    The adaptive damping increment is:
        g_t  = softplus(W_adapt @ [r_t, q_t, c_t] + b_adapt)   shape (P,)

    and the time-varying state matrix is:
        A_t  = Lambda_bar * exp(-g_t)

    Because exp(-g_t) in (0, 1] and |Lambda_bar| <= 1 (S5 stability guarantee),
    |A_t| <= 1 unconditionally.  The update increases damping when the latent
    state looks disrupted (large r_t, q_t >> 1) and reduces it when the state
    is stable/aligned (c_t ≈ 1).

    Args:
        Lambda_bar (complex64): base discretised diagonal state matrix  (P,)
        B_bar      (complex64): discretised input matrix                (P, H)
        C_tilde    (complex64): output matrix                           (H, P)
        W_adapt    (float32):   damping controller weights              (P, 3)
        b_adapt    (float32):   damping controller bias                 (P,)
        input_sequence (float32): input features                        (L, H)
        conj_sym   (bool):     whether conjugate symmetry is enforced
    Returns:
        ys (float32): SSM outputs                                       (L, H)
    """
    P = Lambda_bar.shape[0]
    eps = 1e-6

    def step(carry, u_t):
        h_prev, h_prev2 = carry   # (P,) complex each

        # ---- state diagnostics (all scalar, computed in magnitude space) ----
        diff_norm  = np.linalg.norm(h_prev - h_prev2)
        prev2_norm = np.linalg.norm(h_prev2)
        prev_norm  = np.linalg.norm(h_prev)

        r_t = diff_norm / (prev2_norm + eps)
        q_t = (prev_norm ** 2) / (prev2_norm ** 2 + eps)
        c_t = np.real(np.vdot(h_prev, h_prev2)) / (prev_norm * prev2_norm + eps)

        features = np.array([r_t, q_t, c_t])          # (3,)

        # ---- adaptive damping increment (shape P, real, non-negative) -------
        g_t = jax.nn.softplus(W_adapt @ features + b_adapt)   # (P,)

        # ---- time-varying state matrix: multiplicative decay ----------------
        # |A_t| = |Lambda_bar| * exp(-g_t) <= |Lambda_bar| <= 1
        A_t = Lambda_bar * np.exp(-g_t.astype(Lambda_bar.dtype))  # (P,)

        # ---- state update ---------------------------------------------------
        h_t = A_t * h_prev + B_bar @ u_t.astype(B_bar.dtype)      # (P,)

        # ---- output projection ----------------------------------------------
        if conj_sym:
            y_t = 2 * np.real(C_tilde @ h_t)
        else:
            y_t = np.real(C_tilde @ h_t)

        return (h_t, h_prev), y_t

    h0 = np.zeros(P, dtype=Lambda_bar.dtype)
    init_carry = (h0, h0)

    _, ys = jax.lax.scan(step, init_carry, input_sequence)
    return ys   # (L, H)


class AdaptiveDampingS5SSM(S5SSM):
    """S5 SSM with state-regulated adaptive damping on the diagonal of A_t.

    Inherits all parameters and discretization logic from S5SSM and adds a
    small (P × 3) linear controller that maps three scalar state diagnostics
    (relative change r_t, energy ratio q_t, cosine similarity c_t) to a
    per-state damping increment g_t >= 0.

    The time-varying discrete state matrix at step t is:
        A_t = Lambda_bar * exp(-g_t)
    which is always stable (|A_t| <= |Lambda_bar| <= 1) and recovers the base
    S5 behaviour when g_t -> 0.

    The model uses a sequential JAX scan (jax.lax.scan) rather than the
    associative scan used by S5SSM, because the adaptive damping introduces
    a causal dependency h_{t-1} -> g_t -> A_t that cannot be parallelised.
    It should be presented as a "state-regulated SSM" or
    "adaptive-damping SSM", not as a linear SSM.

    Extra parameters added per layer (beyond S5SSM):
        W_adapt  (float32): controller weight matrix   (P, 3), init zeros
        b_adapt  (float32): controller bias vector     (P,),   init zeros

    Initialising W_adapt = 0 and b_adapt = 0 gives g_t = softplus(0) ≈ 0.693
    uniformly, which acts as a mild fixed extra damping at init.  To start
    closer to vanilla S5, initialise b_adapt to a large negative value
    (e.g. -3) so that softplus(b_adapt) ≈ 0 and the base Lambda_bar
    dominates.
    """

    def setup(self):
        # Inherit all S5SSM parameter setup (Lambda, B, C, D, log_step, discretize)
        super().setup()

        # Adaptive damping controller: maps 3 scalar diagnostics -> P damping increments
        # Initialised so that g_t ≈ 0 at the start (large negative bias).
        self.W_adapt = self.param("W_adapt",
                                  lambda rng, shape: np.zeros(shape),
                                  (self.P, 3))
        self.b_adapt = self.param("b_adapt",
                                  lambda rng, shape: np.full(shape, -3.0),
                                  (self.P,))

    def __call__(self, input_sequence, global_th):
        """
        Compute the LxH output using sequential adaptive-damping scan.
        Args:
            input_sequence (float32): (L, H)
            global_th: pruning threshold (unused; accepted for API compatibility
                       with S5SSM which passes this from layers.py)
        Returns:
            output (float32): (L, H)
            ENERGYscore: None (adaptive damping does not use AIRE scoring)
        """
        ys = apply_ssm_adaptive(
            self.Lambda_bar,
            self.B_bar,
            self.C_tilde,
            self.W_adapt,
            self.b_adapt,
            input_sequence,
            self.conj_sym,
        )
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du, None


def init_AdaptiveDampingS5SSM(H,
                               P,
                               Lambda_re_init,
                               Lambda_im_init,
                               V,
                               Vinv,
                               C_init,
                               discretization,
                               dt_min,
                               dt_max,
                               conj_sym,
                               clip_eigs,
                               bidirectional,
                               pruning=False):
    """Convenience function to initialise AdaptiveDampingS5SSM.
    Same arguments as init_S5SSM; bidirectional and pruning are accepted
    for API compatibility (pruning is passed through but the adaptive model
    returns ENERGYscore=None regardless, since damping is the forgetting
    mechanism)."""
    return partial(AdaptiveDampingS5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional,
                   pruning=pruning)


def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)
