'''Reward-modulated plasticity + structural growth/atrophy on top of the
static connectome model in `model.py`.

This is an experimental extension, not part of the published paper. It keeps
`model.py` untouched and adds a parallel model where synaptic weights are no
longer frozen at their connectome-derived value. Instead:

- Each synapse accumulates an eligibility trace from correlated pre/post
  spiking (classic STDP-style trace), same idea as reward-modulated STDP
  (Izhikevich 2007; Fremaux & Gerstner 2016).
- A scalar "dopamine" reward signal is computed from Python between short
  simulation chunks (whatever a reward function says: did the target
  neuron(s) fire, was a task solved, etc.) and used to push weights up or
  down along their eligibility trace.
- Periodically, synapses that never accumulate eligibility (i.e. never
  correlate with reward) are atrophied toward zero, while synapses pressed
  against their ceiling are given more headroom ("growth"), modelling
  structural strengthening/pruning on top of the fixed anatomical wiring.

Sign of each synapse (excitatory/inhibitory) is preserved: plasticity can
only scale a synapse's strength, not flip its sign, since Flywire only gives
us connection counts and an E/I flag, not detailed receptor biology.
'''

import numpy as np
import pandas as pd
from textwrap import dedent

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz

from model import default_params

plastic_params = dict(default_params)
plastic_params.update({
    # STDP traces used to build the eligibility trace
    'tau_pre'      : 20 * ms,     # presynaptic trace time constant
    'tau_post'     : 20 * ms,     # postsynaptic trace time constant
    'A_pre'        : 0.01,        # presynaptic trace increment per spike
    'A_post'       : 0.01,        # postsynaptic trace increment per spike
    'tau_elig'     : 500 * ms,    # eligibility trace decay (bridges pre/post correlation -> reward)
    'elig_gain'    : 1 * mV,      # volt-scale of trace-driven eligibility increments

    # reward-gated weight update, applied between simulation chunks
    'chunk_dt'     : 50 * ms,     # duration of one reward/plasticity update chunk
    'lr'           : 0.05,        # learning rate applied to elig * dopamine

    # structural growth / atrophy, applied every `review_every` chunks
    'growth_mult'  : 3.0,         # initial headroom = growth_mult * |w_init|
    'w_max_floor'  : 0.5 * mV,    # minimum initial headroom, even for near-zero synapses
    'growth_thr'   : 0.9,         # fraction of ceiling at which a synapse earns more headroom
    'growth_step'  : 1.5,         # multiplicative ceiling growth once growth_thr is reached
    'atrophy_thr'  : 0.02 * mV,   # |eligibility| below this over a review window -> "unused"
    'atrophy_rate' : 0.9,         # multiplicative decay applied to unused synapses' weight
})


def create_plastic_model(path_comp, path_con, params):
    '''Build the connectome model with plastic, reward-eligible synapses.

    Same neuron model as `model.create_model`. Synapses gain an eligibility
    trace (`elig`) built from correlated pre/post spiking, and a per-synapse
    ceiling (`w_max`) that structural growth can raise over time.

    Returns
    -------
    neu, syn, spk_mon : as in `model.create_model`
    df_comp : pandas.DataFrame
        completeness table, kept around for flyid <-> index lookups
    '''

    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    neu = NeuronGroup(
        N=len(df_comp),
        model=params['eqs'],
        method='linear',
        threshold=params['eq_th'],
        reset=params['eq_rst'],
        refractory='rfc',
        name='default_neurons',
        namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']

    syn_model = dedent('''
        w        : volt
        w_max    : volt
        w_sign   : 1
        dapre/dt  = -apre/tau_pre   : 1    (event-driven)
        dapost/dt = -apost/tau_post : 1    (event-driven)
        delig/dt  = -elig/tau_elig  : volt (clock-driven)
        ''')
    syn = Synapses(
        neu, neu, syn_model,
        on_pre='g += w; apre += A_pre; elig += apost*elig_gain',
        on_post='apost += A_post; elig += apre*elig_gain',
        delay=params['t_dly'],
        name='plastic_synapses',
        namespace=params,
    )

    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    # work in plain mV floats to sidestep brian2-Quantity edge cases, then reattach units
    w_init_mV = df_con.loc[:, 'Excitatory x Connectivity'].values * (params['w_syn'] / mV)
    signs = np.sign(w_init_mV)
    signs[signs == 0] = 1
    headroom_mV = np.maximum(np.abs(w_init_mV) * params['growth_mult'], params['w_max_floor'] / mV)

    syn.w = w_init_mV * mV
    syn.w_sign = signs
    syn.w_max = headroom_mV * mV
    syn.apre = 0
    syn.apost = 0
    syn.elig = 0 * mV

    spk_mon = SpikeMonitor(neu)

    return neu, syn, spk_mon, df_comp


def update_weights(syn, dopamine, params):
    '''Reward-gate the eligibility trace into a weight change (one chunk's worth).

    dw = lr * dopamine * eligibility, clipped so a synapse never crosses zero
    (excitatory synapses stay in [0, w_max], inhibitory in [-w_max, 0]).
    '''

    w = np.asarray(syn.w[:] / mV)
    elig = np.asarray(syn.elig[:] / mV)
    wmax = np.asarray(syn.w_max[:] / mV)
    sign = np.asarray(syn.w_sign[:])

    w_new = w + params['lr'] * dopamine * elig

    lo = np.where(sign >= 0, 0.0, -wmax)
    hi = np.where(sign >= 0, wmax, 0.0)
    w_new = np.clip(w_new, lo, hi)

    syn.w = w_new * mV

    return w_new, elig, wmax


def apply_growth_atrophy(syn, params):
    '''Structural update: widen the ceiling for synapses pushing against it,
    decay synapses that never picked up eligibility (never correlated with
    reward) toward zero.
    '''

    w = np.asarray(syn.w[:] / mV)
    wmax = np.asarray(syn.w_max[:] / mV)
    elig = np.asarray(syn.elig[:] / mV)

    near_ceiling = np.abs(w) >= params['growth_thr'] * np.maximum(wmax, 1e-9)
    wmax_new = np.where(near_ceiling, wmax * params['growth_step'], wmax)

    unused = np.abs(elig) < (params['atrophy_thr'] / mV)
    w_new = np.where(unused, w * params['atrophy_rate'], w)

    syn.w = w_new * mV
    syn.w_max = wmax_new * mV

    return float(unused.mean()), float(near_ceiling.mean())


def run_plastic_experiment(path_comp, path_con, neu_exc, neu_reward,
                            params=plastic_params, n_chunks=10, review_every=5,
                            r_poi=None, penalty=-0.2, verbose=True):
    '''Run a short reward/growth pilot on the full connectome.

    Neurons in `neu_exc` receive Poisson input for the whole run (the
    "sensory" side). Every `chunk_dt`, we check whether any neuron in
    `neu_reward` (the "motor"/output side) spiked since the last check; that
    gates a dopamine-style weight update via `update_weights`, and every
    `review_every` chunks `apply_growth_atrophy` runs a structural pass.

    Parameters
    ----------
    neu_exc : list of flywire IDs
        neurons driven with Poisson input (stand-in for a sensory cue)
    neu_reward : list of flywire IDs
        neurons whose firing defines the reward signal (stand-in for a body-part/motor readout)
    n_chunks : int
        number of `params['chunk_dt']`-long chunks to run
    review_every : int
        run `apply_growth_atrophy` every this many chunks
    r_poi : brian2 Hz quantity, optional
        override for the Poisson input rate (defaults to `params['r_poi']`)
    penalty : float
        dopamine value used on chunks where the reward neuron(s) did not fire

    Returns
    -------
    neu, syn, spk_mon : brian2 objects (final state)
    log : list of dict
        one entry per chunk with reward/weight/eligibility/growth stats
    '''

    neu, syn, spk_mon, df_comp = create_plastic_model(path_comp, path_con, params)

    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    exc = [flyid2i[n] for n in neu_exc]
    reward_idx = [flyid2i[n] for n in neu_reward]

    rate = r_poi if r_poi is not None else params['r_poi']
    pois = []
    for i in exc:
        p = PoissonInput(target=neu[i], target_var='v', N=1, rate=rate,
                          weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)

    net = Network(neu, syn, spk_mon, *pois)

    log = []
    prev_reward_spikes = 0
    for c in range(n_chunks):
        net.run(params['chunk_dt'])

        counts = np.asarray(spk_mon.count[:])
        total_reward_spikes = int(counts[reward_idx].sum())
        chunk_reward_spikes = total_reward_spikes - prev_reward_spikes
        prev_reward_spikes = total_reward_spikes

        dopamine = 1.0 if chunk_reward_spikes > 0 else penalty
        w_new, elig, wmax = update_weights(syn, dopamine, params)

        entry = {
            'chunk': c,
            'reward_spikes': chunk_reward_spikes,
            'dopamine': dopamine,
            'total_spikes': int(counts.sum()),
            'mean_abs_w_mV': float(np.mean(np.abs(w_new))),
            'mean_abs_elig_mV': float(np.mean(np.abs(elig))),
            'mean_w_max_mV': float(np.mean(wmax)),
        }

        if (c + 1) % review_every == 0:
            frac_atrophied, frac_grown = apply_growth_atrophy(syn, params)
            entry['frac_atrophied'] = frac_atrophied
            entry['frac_grown'] = frac_grown

        log.append(entry)
        if verbose:
            print(entry)

    return neu, syn, spk_mon, log
