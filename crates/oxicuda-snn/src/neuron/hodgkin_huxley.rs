//! Hodgkin-Huxley multi-compartment conductance-based neuron models.
//!
//! Implements the classic Hodgkin-Huxley (1952) model for a population of
//! neurons, plus the two-compartment Pinsky-Rinzel (1994) CA3 pyramidal cell.
//!
//! # Hodgkin-Huxley (Hodgkin & Huxley 1952, J. Physiol. 117:500)
//!
//! ```text
//! C_m · dV/dt = I_ext − g_Na · m³ · h · (V − E_Na)
//!                     − g_K  · n⁴       · (V − E_K )
//!                     − g_L             · (V − E_L )
//!
//! dx/dt = α_x(V)·(1−x) − β_x(V)·x   for x ∈ {m, h, n}
//! ```
//!
//! Voltage integration uses 4th-order Runge-Kutta; gating variables are
//! updated via exact exponential integration (Rotter & Diesmann 1999).
//!
//! # Pinsky-Rinzel (Pinsky & Rinzel 1994, J. Comput. Neurosci. 1:39)
//!
//! Two-compartment (soma + dendrite) CA3 pyramidal cell model with somatic
//! fast Na/K-DR and dendritic Ca²⁺, Ca²⁺-dependent K, and AHP channels.

use crate::error::{SnnError, SnnResult};

// ── Hodgkin-Huxley ────────────────────────────────────────────────────────────

/// Standard HH biophysical parameters (squid-axon defaults).
#[derive(Debug, Clone, Copy)]
pub struct HhConfig {
    /// Integration time step ms (default 0.01).
    pub dt: f32,
    /// Membrane capacitance µF/cm² (default 1.0).
    pub c_m: f32,
    /// Max Na conductance mS/cm² (default 120.0).
    pub g_na: f32,
    /// Max K conductance mS/cm² (default 36.0).
    pub g_k: f32,
    /// Leak conductance mS/cm² (default 0.3).
    pub g_l: f32,
    /// Na reversal potential mV (default 50.0).
    pub e_na: f32,
    /// K reversal potential mV (default −77.0).
    pub e_k: f32,
    /// Leak reversal potential mV (default −54.387).
    pub e_l: f32,
    /// Resting membrane potential mV (default −65.0).
    pub v_rest: f32,
}

impl Default for HhConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            v_rest: -65.0,
        }
    }
}

/// Per-neuron HH state: membrane voltage and three gating variables.
#[derive(Debug, Clone)]
pub struct HhState {
    /// Membrane potential mV, length `n`.
    pub v: Vec<f32>,
    /// Na activation gating variable ∈ `[0,1]`, length `n`.
    pub m: Vec<f32>,
    /// Na inactivation gating variable ∈ `[0,1]`, length `n`.
    pub h: Vec<f32>,
    /// K activation gating variable ∈ `[0,1]`, length `n`.
    pub n: Vec<f32>,
    /// Binary spike output (1.0 if V crossed 0 mV from below), length `n`.
    pub spikes: Vec<f32>,
}

impl HhState {
    /// Allocate state for `n_neurons` neurons initialised at `v_rest` with
    /// gating variables set to their steady-state values at `v_rest`.
    #[must_use]
    pub fn new(n_neurons: usize, cfg: &HhConfig) -> Self {
        let v_rest = cfg.v_rest;
        let (m_inf, h_inf, n_inf) = gating_steady_state(v_rest);
        Self {
            v: vec![v_rest; n_neurons],
            m: vec![m_inf; n_neurons],
            h: vec![h_inf; n_neurons],
            n: vec![n_inf; n_neurons],
            spikes: vec![0.0_f32; n_neurons],
        }
    }

    /// Construct from explicit initial vectors (must all have equal length > 0).
    pub fn new_custom(
        v_init: Vec<f32>,
        m_init: Vec<f32>,
        h_init: Vec<f32>,
        n_init: Vec<f32>,
    ) -> SnnResult<Self> {
        let len = v_init.len();
        if len == 0 {
            return Err(SnnError::EmptyInput);
        }
        if m_init.len() != len {
            return Err(SnnError::IncompatibleLength {
                a: len,
                b: m_init.len(),
            });
        }
        if h_init.len() != len {
            return Err(SnnError::IncompatibleLength {
                a: len,
                b: h_init.len(),
            });
        }
        if n_init.len() != len {
            return Err(SnnError::IncompatibleLength {
                a: len,
                b: n_init.len(),
            });
        }
        Ok(Self {
            spikes: vec![0.0_f32; len],
            v: v_init,
            m: m_init,
            h: h_init,
            n: n_init,
        })
    }
}

// ── HH rate functions ─────────────────────────────────────────────────────────

/// α_m: Na activation opening rate. Handles the limit at V = −40 mV.
#[inline]
fn alpha_m(v: f32) -> f32 {
    let x = v + 40.0;
    if x.abs() < 1e-5 {
        // L'Hôpital limit: lim_{x→0} 0.1·x / (1 − e^{−x/10}) = 1.0
        1.0
    } else {
        0.1 * x / (1.0 - (-x / 10.0).exp())
    }
}

/// β_m: Na activation closing rate.
#[inline]
fn beta_m(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

/// α_h: Na inactivation opening rate.
#[inline]
fn alpha_h(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}

/// β_h: Na inactivation closing rate.
#[inline]
fn beta_h(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

/// α_n: K activation opening rate. Handles the limit at V = −55 mV.
#[inline]
fn alpha_n(v: f32) -> f32 {
    let x = v + 55.0;
    if x.abs() < 1e-5 {
        // L'Hôpital limit: lim_{x→0} 0.01·x / (1 − e^{−x/10}) = 0.1
        0.1
    } else {
        0.01 * x / (1.0 - (-x / 10.0).exp())
    }
}

/// β_n: K activation closing rate.
#[inline]
fn beta_n(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// Compute (m_inf, h_inf, n_inf) at a given voltage.
fn gating_steady_state(v: f32) -> (f32, f32, f32) {
    let am = alpha_m(v);
    let bm = beta_m(v);
    let ah = alpha_h(v);
    let bh = beta_h(v);
    let an = alpha_n(v);
    let bn = beta_n(v);
    (am / (am + bm), ah / (ah + bh), an / (an + bn))
}

/// Exact exponential update for a single gating variable at voltage `v`.
/// Uses: x(t+dt) = x_inf + (x − x_inf)·exp(−dt/τ_x)
#[inline]
fn gate_exact(x: f32, alpha: f32, beta: f32, dt: f32) -> f32 {
    let tau = 1.0 / (alpha + beta);
    let x_inf = alpha * tau;
    let updated = x_inf + (x - x_inf) * (-dt / tau).exp();
    updated.clamp(0.0, 1.0)
}

/// Ionic current (without sign flip): I_ion = g_Na·m³·h·(V−E_Na) + g_K·n⁴·(V−E_K) + g_L·(V−E_L)
#[inline]
fn i_ion(v: f32, m: f32, h: f32, n_g: f32, cfg: &HhConfig) -> f32 {
    let i_na = cfg.g_na * m * m * m * h * (v - cfg.e_na);
    let i_k = cfg.g_k * n_g * n_g * n_g * n_g * (v - cfg.e_k);
    let i_l = cfg.g_l * (v - cfg.e_l);
    i_na + i_k + i_l
}

/// dV/dt = (I_ext − I_ion) / C_m
#[inline]
fn dv_dt(v: f32, m: f32, h: f32, n_g: f32, i_ext: f32, cfg: &HhConfig) -> f32 {
    (i_ext - i_ion(v, m, h, n_g, cfg)) / cfg.c_m
}

/// Validate `cfg` and slice lengths for `hh_step`.
fn validate_hh(state: &HhState, current: &[f32], cfg: &HhConfig) -> SnnResult<()> {
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    let n = state.v.len();
    if n == 0 {
        return Err(SnnError::EmptyInput);
    }
    if current.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: current.len(),
        });
    }
    Ok(())
}

/// Advance a population of HH neurons by one timestep.
///
/// Uses RK4 for the membrane voltage `V` (with frozen gating during sub-steps)
/// and exact exponential integration for the gating variables evaluated at the
/// post-RK4 voltage.  Writes 1.0 to `state.spikes[i]` if neuron `i` crossed
/// 0 mV from below (i.e. V_old < 0 ≤ V_new).
pub fn hh_step(state: &mut HhState, current: &[f32], cfg: &HhConfig) -> SnnResult<()> {
    validate_hh(state, current, cfg)?;
    let dt = cfg.dt;

    // Iterate all five per-neuron buffers in lockstep.
    let iter = state
        .v
        .iter_mut()
        .zip(state.m.iter_mut())
        .zip(state.h.iter_mut())
        .zip(state.n.iter_mut())
        .zip(state.spikes.iter_mut())
        .zip(current.iter());

    for (((((v_ref, m_ref), h_ref), n_ref), spike_ref), &i_ext) in iter {
        let v_old = *v_ref;
        let m_t = *m_ref;
        let h_t = *h_ref;
        let n_t = *n_ref;

        // RK4 on V (gating vars held at time-t values for all stages)
        let k1 = dv_dt(v_old, m_t, h_t, n_t, i_ext, cfg);
        let k2 = dv_dt(v_old + 0.5 * dt * k1, m_t, h_t, n_t, i_ext, cfg);
        let k3 = dv_dt(v_old + 0.5 * dt * k2, m_t, h_t, n_t, i_ext, cfg);
        let k4 = dv_dt(v_old + dt * k3, m_t, h_t, n_t, i_ext, cfg);
        let v_new = v_old + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        // Exact exponential gating update at the new voltage
        let am = alpha_m(v_new);
        let bm = beta_m(v_new);
        let ah = alpha_h(v_new);
        let bh = beta_h(v_new);
        let an = alpha_n(v_new);
        let bn = beta_n(v_new);

        *m_ref = gate_exact(m_t, am, bm, dt);
        *h_ref = gate_exact(h_t, ah, bh, dt);
        *n_ref = gate_exact(n_t, an, bn, dt);

        // Spike: threshold crossing of 0 mV from below
        *spike_ref = if v_old < 0.0 && v_new >= 0.0 {
            1.0
        } else {
            0.0
        };
        *v_ref = v_new;
    }
    Ok(())
}

/// Run `currents.len()` timesteps and collect spike rasters.
///
/// `currents[t]` must have the same length as `state.v` for every `t`.
/// Returns `raster[t][i]` = 1.0 if neuron `i` spiked at step `t`.
pub fn hh_run(
    state: &mut HhState,
    currents: &[Vec<f32>],
    cfg: &HhConfig,
) -> SnnResult<Vec<Vec<f32>>> {
    if currents.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    let n = state.v.len();
    if n == 0 {
        return Err(SnnError::EmptyInput);
    }
    let mut raster: Vec<Vec<f32>> = Vec::with_capacity(currents.len());
    for current in currents {
        hh_step(state, current, cfg)?;
        raster.push(state.spikes.clone());
    }
    Ok(raster)
}

// ── Pinsky-Rinzel two-compartment model ───────────────────────────────────────

/// Two-compartment Pinsky-Rinzel config (CA3 pyramidal cell defaults).
#[derive(Debug, Clone, Copy)]
pub struct PrConfig {
    /// Integration time step ms (default 0.1).
    pub dt: f32,
    /// Membrane capacitance µF/cm² (default 3.0).
    pub c_m: f32,
    /// Somatic-dendritic coupling conductance mS/cm² (default 2.1).
    pub gc: f32,
    /// Fraction of area in somatic compartment ∈ (0,1) (default 0.5).
    /// The dendritic fraction is `(1 − p)`.
    pub p: f32,
    /// Somatic Na conductance mS/cm² (default 60.0).
    pub g_na_s: f32,
    /// Somatic K-DR conductance mS/cm² (default 10.0).
    pub g_k_s: f32,
    /// Somatic leak conductance mS/cm² (default 0.1).
    pub g_l_s: f32,
    /// Dendritic Na conductance mS/cm² (default 0.0).
    pub g_na_d: f32,
    /// AHP K conductance mS/cm² (default 0.8).
    pub g_k_ahp: f32,
    /// Ca²⁺-dependent K conductance mS/cm² (default 15.0).
    pub g_k_c: f32,
    /// Ca²⁺ conductance mS/cm² (default 10.0).
    pub g_ca: f32,
    /// Dendritic leak conductance mS/cm² (default 0.1).
    pub g_l_d: f32,
    /// Na reversal potential mV (default 60.0).
    pub e_na: f32,
    /// K reversal potential mV (default −75.0).
    pub e_k: f32,
    /// Leak reversal potential mV (default −60.0).
    pub e_l: f32,
    /// Ca²⁺ reversal potential mV (default 80.0).
    pub e_ca: f32,
}

impl Default for PrConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,
            c_m: 3.0,
            gc: 2.1,
            p: 0.5,
            g_na_s: 60.0,
            g_k_s: 10.0,
            g_l_s: 0.1,
            g_na_d: 0.0,
            g_k_ahp: 0.8,
            g_k_c: 15.0,
            g_ca: 10.0,
            g_l_d: 0.1,
            e_na: 60.0,
            e_k: -75.0,
            e_l: -60.0,
            e_ca: 80.0,
        }
    }
}

/// Per-neuron Pinsky-Rinzel state (all vectors have equal length `n`).
#[derive(Debug, Clone)]
pub struct PrState {
    /// Somatic membrane potential mV.
    pub v_s: Vec<f32>,
    /// Dendritic membrane potential mV.
    pub v_d: Vec<f32>,
    /// Somatic Na activation gating variable.
    pub m_na: Vec<f32>,
    /// Somatic Na inactivation gating variable.
    pub h_na: Vec<f32>,
    /// Somatic K-DR activation gating variable.
    pub n_k: Vec<f32>,
    /// Dendritic Ca²⁺ channel activation (slow NMDA-like gating).
    pub s_ca: Vec<f32>,
    /// Intracellular Ca²⁺ concentration (arbitrary units).
    pub ca: Vec<f32>,
    /// AHP K activation (slow Ca²⁺-driven).
    pub q_ahp: Vec<f32>,
    /// Ca²⁺-dependent K activation (instantaneous, stored for output).
    pub c_kc: Vec<f32>,
}

impl PrState {
    /// Allocate state for `n_neurons` neurons initialised at −60 mV,
    /// all gating variables at 0.
    #[must_use]
    pub fn new(n_neurons: usize) -> Self {
        Self {
            v_s: vec![-60.0_f32; n_neurons],
            v_d: vec![-60.0_f32; n_neurons],
            m_na: vec![0.0_f32; n_neurons],
            h_na: vec![0.0_f32; n_neurons],
            n_k: vec![0.0_f32; n_neurons],
            s_ca: vec![0.0_f32; n_neurons],
            ca: vec![0.0_f32; n_neurons],
            q_ahp: vec![0.0_f32; n_neurons],
            c_kc: vec![0.0_f32; n_neurons],
        }
    }
}

/// Somatic Na activation α (same HH-style formula shifted to PR voltage).
#[inline]
fn pr_alpha_m(v: f32) -> f32 {
    let x = v + 40.0;
    if x.abs() < 1e-5 {
        1.0
    } else {
        0.1 * x / (1.0 - (-x / 10.0).exp())
    }
}

#[inline]
fn pr_beta_m(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

#[inline]
fn pr_alpha_h(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}

#[inline]
fn pr_beta_h(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

#[inline]
fn pr_alpha_n(v: f32) -> f32 {
    let x = v + 55.0;
    if x.abs() < 1e-5 {
        0.1
    } else {
        0.01 * x / (1.0 - (-x / 10.0).exp())
    }
}

#[inline]
fn pr_beta_n(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// Simplified dendritic Ca²⁺ channel: use logistic for s_ca steady-state approach.
/// α_s(V) / (α_s(V) + β_s(V)) evaluated more stably.
#[inline]
fn pr_s_inf(v: f32) -> f32 {
    1.0 / (1.0 + (-0.062 * v).exp() * (1.0 / 3.736))
}

/// Single Euler step for the Pinsky-Rinzel two-compartment model.
///
/// `i_s` = somatic injection current µA/cm², `i_d` = dendritic injection,
/// both of length `n`.
pub fn pr_step(state: &mut PrState, i_s: &[f32], i_d: &[f32], cfg: &PrConfig) -> SnnResult<()> {
    let n = state.v_s.len();
    if n == 0 {
        return Err(SnnError::EmptyInput);
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if i_s.len() != n {
        return Err(SnnError::IncompatibleLength { a: n, b: i_s.len() });
    }
    if i_d.len() != n {
        return Err(SnnError::IncompatibleLength { a: n, b: i_d.len() });
    }

    let dt = cfg.dt;
    let p = cfg.p.clamp(1e-4, 1.0 - 1e-4);

    for i in 0..n {
        let vs = state.v_s[i];
        let vd = state.v_d[i];
        let m = state.m_na[i];
        let h = state.h_na[i];
        let nk = state.n_k[i];
        let s = state.s_ca[i];
        let ca = state.ca[i];
        let q = state.q_ahp[i];

        // Coupling currents
        let i_coup_s = cfg.gc / p * (vd - vs);
        let i_coup_d = cfg.gc / (1.0 - p) * (vs - vd);

        // Somatic ionic currents
        let i_na_s = cfg.g_na_s * m * m * h * (vs - cfg.e_na);
        let i_k_s = cfg.g_k_s * nk * nk * nk * nk * (vs - cfg.e_k);
        let i_l_s = cfg.g_l_s * (vs - cfg.e_l);

        // Dendritic ionic currents
        let i_ca = cfg.g_ca * s * s * (vd - cfg.e_ca);
        // c_kc: instantaneous Ca²⁺-dependent K activation
        let c_kc_val = (ca / 250.0).min(1.0);
        let i_k_c = cfg.g_k_c * c_kc_val * (vd - cfg.e_k);
        let i_k_ahp = cfg.g_k_ahp * q * (vd - cfg.e_k);
        let i_l_d = cfg.g_l_d * (vd - cfg.e_l);
        let i_na_d = cfg.g_na_d * (vd - cfg.e_na); // usually 0

        // Membrane potential derivatives (Euler)
        let dvs_dt = (i_s[i] - i_na_s - i_k_s - i_l_s + i_coup_s) / (cfg.c_m * p);
        let dvd_dt =
            (i_d[i] - i_ca - i_k_c - i_k_ahp - i_l_d - i_na_d + i_coup_d) / (cfg.c_m * (1.0 - p));

        // Gating variable derivatives (Euler on somatic HH gates)
        let am = pr_alpha_m(vs);
        let bm = pr_beta_m(vs);
        let ah = pr_alpha_h(vs);
        let bh = pr_beta_h(vs);
        let an = pr_alpha_n(vs);
        let bn = pr_beta_n(vs);

        let dm_dt = am * (1.0 - m) - bm * m;
        let dh_dt = ah * (1.0 - h) - bh * h;
        let dn_dt = an * (1.0 - nk) - bn * nk;

        // Dendritic Ca²⁺ slow gate (towards s_inf with time constant ~80 ms)
        let s_inf = pr_s_inf(vd);
        let tau_s = 80.0;
        let ds_dt = (s_inf - s) / tau_s;

        // Ca²⁺ dynamics: dCa/dt = −0.13·I_Ca − 0.075·Ca
        let dca_dt = -0.13 * i_ca - 0.075 * ca;

        // AHP gate: dq/dt = min(0.00002·Ca, 0.01) − 0.001·q
        let dq_dt = (0.00002 * ca).min(0.01) - 0.001 * q;

        // Euler updates
        state.v_s[i] = vs + dt * dvs_dt;
        state.v_d[i] = vd + dt * dvd_dt;
        state.m_na[i] = (m + dt * dm_dt).clamp(0.0, 1.0);
        state.h_na[i] = (h + dt * dh_dt).clamp(0.0, 1.0);
        state.n_k[i] = (nk + dt * dn_dt).clamp(0.0, 1.0);
        state.s_ca[i] = (s + dt * ds_dt).clamp(0.0, 1.0);
        state.ca[i] = (ca + dt * dca_dt).max(0.0);
        state.q_ahp[i] = (q + dt * dq_dt).clamp(0.0, 1.0);
        state.c_kc[i] = c_kc_val;
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> HhConfig {
        HhConfig::default()
    }

    // 1. Default config + 1 step with I=10, no error
    #[test]
    fn hh_default_config_step_no_panic() {
        let cfg = default_cfg();
        let mut state = HhState::new(1, &cfg);
        let current = vec![10.0_f32];
        hh_step(&mut state, &current, &cfg).expect("step should succeed");
    }

    // 2. Zero current → no spikes, V near v_rest
    #[test]
    fn hh_resting_no_spike_zero_current() {
        let cfg = default_cfg();
        let mut state = HhState::new(1, &cfg);
        let current = vec![0.0_f32];
        let mut total_spikes = 0.0_f32;
        for _ in 0..1000 {
            hh_step(&mut state, &current, &cfg).expect("step");
            total_spikes += state.spikes[0];
        }
        assert_eq!(total_spikes, 0.0, "no spikes expected at rest");
        assert!(
            (state.v[0] - cfg.v_rest).abs() < 5.0,
            "V should remain near v_rest, got {}",
            state.v[0]
        );
    }

    // 3. Large current (I=20 µA/cm²) → at least 1 spike in 1000 steps (10 ms at dt=0.01)
    #[test]
    fn hh_spike_with_large_current() {
        let cfg = default_cfg(); // dt=0.01 ms
        let mut state = HhState::new(1, &cfg);
        let current = vec![20.0_f32];
        let mut total_spikes = 0.0_f32;
        // 1000 steps × 0.01 ms = 10 ms, ample time for HH to spike at I=20 µA/cm²
        for _ in 0..1000 {
            hh_step(&mut state, &current, &cfg).expect("step");
            total_spikes += state.spikes[0];
        }
        assert!(
            total_spikes >= 1.0,
            "expected at least 1 spike in 10 ms with I=20 µA/cm², got {}",
            total_spikes
        );
    }

    // 4. m_inf at v_rest ≈ 0.0529
    #[test]
    fn hh_m_steady_state() {
        let cfg = default_cfg();
        let v = cfg.v_rest;
        let am = alpha_m(v);
        let bm = beta_m(v);
        let m_inf = am / (am + bm);
        assert!(
            (m_inf - 0.0529).abs() < 1e-3,
            "m_inf at v_rest = {}, expected ≈ 0.0529",
            m_inf
        );
    }

    // 5. h_inf at v_rest ≈ 0.596
    #[test]
    fn hh_h_steady_state() {
        let cfg = default_cfg();
        let v = cfg.v_rest;
        let ah = alpha_h(v);
        let bh = beta_h(v);
        let h_inf = ah / (ah + bh);
        assert!(
            (h_inf - 0.596).abs() < 1e-2,
            "h_inf at v_rest = {}, expected ≈ 0.596",
            h_inf
        );
    }

    // 6. n_inf at v_rest ≈ 0.318 (HH 1952 squid axon)
    #[test]
    fn hh_n_steady_state() {
        let cfg = default_cfg();
        let v = cfg.v_rest;
        let an = alpha_n(v);
        let bn = beta_n(v);
        let n_inf = an / (an + bn);
        // Expected value is ~0.318; use tolerance-based check against 0.32.
        let target: f32 = 32e-2; // 0.32
        assert!(
            (n_inf - target).abs() < 5e-3,
            "n_inf at v_rest = {}, expected ≈ 0.318",
            n_inf
        );
    }

    // 7. Gating vars always in [0,1] after 500 steps
    #[test]
    fn hh_gating_clamped_to_unit_interval() {
        let cfg = default_cfg();
        let mut state = HhState::new(1, &cfg);
        let current = vec![30.0_f32];
        for _ in 0..500 {
            hh_step(&mut state, &current, &cfg).expect("step");
            assert!(
                (0.0..=1.0).contains(&state.m[0]),
                "m out of range: {}",
                state.m[0]
            );
            assert!(
                (0.0..=1.0).contains(&state.h[0]),
                "h out of range: {}",
                state.h[0]
            );
            assert!(
                (0.0..=1.0).contains(&state.n[0]),
                "n out of range: {}",
                state.n[0]
            );
        }
    }

    // 8. Multiple neurons with different currents evolve independently
    #[test]
    fn hh_multi_neuron_independent() {
        let cfg = default_cfg();
        // Run 4 neurons together
        let mut state4 = HhState::new(4, &cfg);
        let current4 = vec![5.0_f32, 10.0, 15.0, 20.0];
        for _ in 0..200 {
            hh_step(&mut state4, &current4, &cfg).expect("step");
        }

        // Run each neuron individually and compare
        let currents_single = [5.0_f32, 10.0, 15.0, 20.0];
        for (idx, &c) in currents_single.iter().enumerate() {
            let mut state1 = HhState::new(1, &cfg);
            let cur = vec![c];
            for _ in 0..200 {
                hh_step(&mut state1, &cur, &cfg).expect("step");
            }
            assert!(
                (state4.v[idx] - state1.v[0]).abs() < 1e-3,
                "neuron {} v mismatch: multi={} single={}",
                idx,
                state4.v[idx],
                state1.v[0]
            );
        }
    }

    // 9. hh_run returns correct shape
    #[test]
    fn hh_run_returns_correct_shape() {
        let cfg = default_cfg();
        let n_neurons = 3_usize;
        let t_steps = 50_usize;
        let mut state = HhState::new(n_neurons, &cfg);
        let currents: Vec<Vec<f32>> = (0..t_steps).map(|_| vec![10.0_f32; n_neurons]).collect();
        let raster = hh_run(&mut state, &currents, &cfg).expect("run");
        assert_eq!(raster.len(), t_steps, "raster length mismatch");
        for row in &raster {
            assert_eq!(row.len(), n_neurons, "row length mismatch");
        }
    }

    // 10. Population firing rate > 0 with I=20, 200 ms / dt=0.01 → 20000 steps
    #[test]
    fn hh_population_firing_rate() {
        let cfg = default_cfg();
        let n = 100_usize;
        let mut state = HhState::new(n, &cfg);
        let t_steps = 20_000_usize; // 200 ms
        let currents: Vec<Vec<f32>> = (0..t_steps).map(|_| vec![20.0_f32; n]).collect();
        let raster = hh_run(&mut state, &currents, &cfg).expect("run");
        let total: f32 = raster.iter().flat_map(|row| row.iter().copied()).sum();
        assert!(total > 0.0, "expected spikes in 200 ms with I=20 µA/cm²");
    }

    // 11. Empty current slice → Err
    #[test]
    fn hh_err_empty_state() {
        let cfg = default_cfg();
        let mut state = HhState::new(0, &cfg);
        let current: Vec<f32> = vec![];
        let err = hh_step(&mut state, &current, &cfg);
        assert!(matches!(err, Err(SnnError::EmptyInput)));
    }

    // 12. Negative dt → Err
    #[test]
    fn hh_err_negative_dt() {
        let cfg = HhConfig {
            dt: -0.01,
            ..HhConfig::default()
        };
        let mut state = HhState::new(1, &HhConfig::default());
        let current = vec![0.0_f32];
        let err = hh_step(&mut state, &current, &cfg);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    // 13. Length mismatch → Err
    #[test]
    fn hh_err_length_mismatch() {
        let cfg = default_cfg();
        let mut state = HhState::new(3, &cfg);
        let current = vec![0.0_f32; 2]; // wrong length
        let err = hh_step(&mut state, &current, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    // 14. α_m(-40.0) = 1.0 (L'Hôpital limit)
    #[test]
    fn hh_alpha_m_limit_at_minus40() {
        let am = alpha_m(-40.0);
        assert!((am - 1.0).abs() < 1e-5, "α_m(-40) = {}, expected 1.0", am);
    }

    // 15. α_n(-55.0) = 0.1 (L'Hôpital limit)
    #[test]
    fn hh_alpha_n_limit_at_minus55() {
        let an = alpha_n(-55.0);
        assert!((an - 0.1).abs() < 1e-5, "α_n(-55) = {}, expected 0.1", an);
    }

    // 16. PR step with zero currents — no panic
    #[test]
    fn pr_step_no_panic() {
        let cfg = PrConfig::default();
        let mut state = PrState::new(2);
        let i_s = vec![0.0_f32; 2];
        let i_d = vec![0.0_f32; 2];
        pr_step(&mut state, &i_s, &i_d, &cfg).expect("pr_step should succeed");
    }

    // 17. Non-zero dendritic injection changes v_d
    #[test]
    fn pr_dendritic_injection_changes_vd() {
        let cfg = PrConfig::default();
        let mut state = PrState::new(1);
        let v_d_before = state.v_d[0];
        let i_s = vec![0.0_f32];
        let i_d = vec![10.0_f32];
        pr_step(&mut state, &i_s, &i_d, &cfg).expect("step");
        assert!(
            (state.v_d[0] - v_d_before).abs() > 1e-6,
            "v_d should change with non-zero i_d"
        );
    }

    // 18. Equal V_s = V_d, no injection → coupling term is zero, voltages equal
    #[test]
    fn pr_coupling_equilibrates() {
        let cfg = PrConfig::default();
        let mut state = PrState::new(1);
        // Set both compartments to identical voltage
        state.v_s[0] = -60.0;
        state.v_d[0] = -60.0;
        let i_s = vec![0.0_f32];
        let i_d = vec![0.0_f32];
        pr_step(&mut state, &i_s, &i_d, &cfg).expect("step");
        // When V_s == V_d, the coupling terms cancel; voltages remain equal
        let diff = (state.v_s[0] - state.v_d[0]).abs();
        assert!(
            diff < 1e-4,
            "v_s and v_d should remain equal when starting equal, diff={}",
            diff
        );
    }
}
