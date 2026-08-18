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

    # neurogenesis: spawn a new neuron when an existing one is saturated (rewarded,
    # but out of headroom to grow further) for several reviews in a row
    'neurogenesis'    : True,
    'sat_frac_thr'    : 0.6,      # fraction of a neuron's outgoing synapses near ceiling to count it as "saturated" this review
    'sat_patience'    : 3,        # consecutive saturated reviews before a neuron earns a duplicate
    'duplicate_scale' : 0.5,      # new neuron's copied synapses start at this fraction of the parent's weight/ceiling
    'max_new_neurons' : 20,       # hard cap on neurons grown in one experiment run

    # efficiency: reward isn't just "did the target fire" -- it's also gated by
    # how fast it fired and how much tissue/activity it cost to get there, so
    # plasticity (and neurogenesis) are pushed toward cheap, fast solutions
    # instead of just firing/growing everything harder
    'penalty'        : -0.2,      # dopamine when the reward neuron(s) don't fire this chunk
    'speed_gain'     : 0.5,       # max bonus for the reward neuron firing right at the start of the chunk (decays to 0 by chunk end)
    'active_syn_thr' : 0.05 * mV, # |w| above this counts as an "active" (resource-costing) synapse
    'cost_spike'     : 2e-4,      # dopamine penalty per network-wide spike this chunk
    'cost_synapse'   : 2e-5,      # dopamine penalty per active synapse this chunk
    'cost_neuron'    : 0.05,      # dopamine penalty per neuron grown beyond the original connectome
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


def snapshot_state(neu, syn):
    '''Pull a live network's full state into plain numpy arrays.

    This is the hand-off point for neurogenesis: Brian2 can't resize a
    `NeuronGroup`/`Synapses` in place, so growing the network means capturing
    everything here, building a bigger network, and restoring it below.
    '''

    return {
        'v'      : np.asarray(neu.v[:] / mV),
        'g'      : np.asarray(neu.g[:] / mV),
        'rfc'    : np.asarray(neu.rfc[:] / ms),
        'i'      : np.asarray(syn.i[:]),
        'j'      : np.asarray(syn.j[:]),
        'w'      : np.asarray(syn.w[:] / mV),
        'w_max'  : np.asarray(syn.w_max[:] / mV),
        'w_sign' : np.asarray(syn.w_sign[:]),
        'apre'   : np.asarray(syn.apre[:]),
        'apost'  : np.asarray(syn.apost[:]),
        'elig'   : np.asarray(syn.elig[:] / mV),
    }


def find_growth_candidates(state, sat_streak, params):
    '''Decide which existing neurons have earned a duplicate this review.

    A neuron is "saturated" this review if most of its outgoing synapses are
    pinned near their ceiling (i.e. reward keeps pushing them up but there's
    no more headroom). `sat_streak` counts consecutive saturated reviews per
    neuron; once a neuron crosses `sat_patience` it is returned as a growth
    candidate and its streak resets.

    Parameters
    ----------
    state : dict
        output of `snapshot_state`
    sat_streak : np.ndarray
        persistent per-neuron counter, updated in place

    Returns
    -------
    candidates : list of int
        indices of neurons to duplicate this review
    '''

    n_neu = len(state['v'])
    near_ceiling = np.abs(state['w']) >= params['growth_thr'] * np.maximum(state['w_max'], 1e-9)

    out_count = np.bincount(state['i'], minlength=n_neu)
    out_sat = np.bincount(state['i'], weights=near_ceiling.astype(float), minlength=n_neu)
    frac_sat = np.divide(out_sat, out_count, out=np.zeros(n_neu), where=out_count > 0)

    saturated = frac_sat >= params['sat_frac_thr']
    sat_streak[saturated] += 1
    sat_streak[~saturated] = 0

    candidates = list(np.where(sat_streak >= params['sat_patience'])[0])
    for idx in candidates:
        sat_streak[idx] = 0

    return candidates


def rebuild_model_with_growth(state, params, parents):
    '''Rebuild the network one size larger, duplicating each neuron in `parents`.

    Each new neuron is spliced in as a partial copy of its parent: it inherits
    a scaled-down (`duplicate_scale`) version of the parent's incoming and
    outgoing synapses, and starts at resting potential. All pre-existing
    neuron/synapse state is restored exactly (this is the actual "growth"
    step -- everything before this call ran on the old, smaller network).

    Parameters
    ----------
    state : dict
        output of `snapshot_state`, taken from the network being grown
    parents : list of int
        neuron indices to duplicate

    Returns
    -------
    neu, syn, spk_mon : brian2 objects for the new, larger network
    new_indices : list of int
        indices assigned to the newly grown neurons (in `parents` order)
    '''

    n_old = len(state['v'])
    n_new = len(parents)
    n_total = n_old + n_new
    scale = params['duplicate_scale']

    neu = NeuronGroup(
        N=n_total, model=params['eqs'], method='linear',
        threshold=params['eq_th'], reset=params['eq_rst'],
        refractory='rfc', name='default_neurons', namespace=params,
    )
    neu.v[:n_old] = state['v'] * mV
    neu.g[:n_old] = state['g'] * mV
    neu.rfc[:n_old] = state['rfc'] * ms
    neu.v[n_old:] = params['v_0']
    neu.g[n_old:] = 0 * mV
    neu.rfc[n_old:] = params['t_rfc']

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

    i_parts  = [state['i']]
    j_parts  = [state['j']]
    w_parts  = [state['w']]
    wm_parts = [state['w_max']]
    sg_parts = [state['w_sign']]
    ap_parts = [state['apre']]
    ao_parts = [state['apost']]
    el_parts = [state['elig']]

    new_indices = []
    for k, parent in enumerate(parents):
        new_idx = n_old + k
        new_indices.append(new_idx)

        out_mask = state['i'] == parent
        in_mask = state['j'] == parent
        n_out = int(out_mask.sum())
        n_in = int(in_mask.sum())

        # outgoing: new neuron -> parent's post-synaptic targets
        i_parts.append(np.full(n_out, new_idx))
        j_parts.append(state['j'][out_mask])
        w_parts.append(state['w'][out_mask] * scale)
        wm_parts.append(state['w_max'][out_mask] * scale)
        sg_parts.append(state['w_sign'][out_mask])
        ap_parts.append(np.zeros(n_out))
        ao_parts.append(np.zeros(n_out))
        el_parts.append(np.zeros(n_out))

        # incoming: parent's pre-synaptic sources -> new neuron
        i_parts.append(state['i'][in_mask])
        j_parts.append(np.full(n_in, new_idx))
        w_parts.append(state['w'][in_mask] * scale)
        wm_parts.append(state['w_max'][in_mask] * scale)
        sg_parts.append(state['w_sign'][in_mask])
        ap_parts.append(np.zeros(n_in))
        ao_parts.append(np.zeros(n_in))
        el_parts.append(np.zeros(n_in))

    syn.connect(i=np.concatenate(i_parts), j=np.concatenate(j_parts))
    syn.w = np.concatenate(w_parts) * mV
    syn.w_max = np.concatenate(wm_parts) * mV
    syn.w_sign = np.concatenate(sg_parts)
    syn.apre = np.concatenate(ap_parts)
    syn.apost = np.concatenate(ao_parts)
    syn.elig = np.concatenate(el_parts) * mV

    spk_mon = SpikeMonitor(neu)

    return neu, syn, spk_mon, new_indices


def _reward_latency_ms(spk_mon, reward_idx, chunk_start_ms):
    '''Milliseconds from the start of the current chunk to the first spike
    among `reward_idx` neurons, or None if none fired.'''

    t = np.asarray(spk_mon.t[:] / ms)
    i = np.asarray(spk_mon.i[:])
    mask = (t >= chunk_start_ms - 1e-6) & np.isin(i, reward_idx)
    if not mask.any():
        return None
    return float(t[mask].min() - chunk_start_ms)


def compute_efficiency_reward(chunk_reward_spikes, latency_ms, chunk_total_spikes,
                               n_active_synapses, n_grown_neurons, params):
    '''Combine task success, response speed, and resource cost into one dopamine value.

    - task success: +1 this chunk if the reward neuron(s) fired, else `params['penalty']`
    - speed bonus: extra reward the earlier in the chunk the first reward spike landed
      (an instant response gets the full `speed_gain`, one at the very end of the
      chunk gets ~0) -- rewards a fast circuit, not just an eventually-correct one
    - resource cost: penalizes total network-wide spiking, the number of synapses
      still "active" (not atrophied), and neurons grown beyond the original
      connectome -- so a solution that fires everything harder or keeps growing
      indefinitely is worth less than a lean one that achieves the same task success
    '''

    task = 1.0 if chunk_reward_spikes > 0 else params['penalty']

    speed = 0.0
    if chunk_reward_spikes > 0 and latency_ms is not None:
        chunk_dt_ms = params['chunk_dt'] / ms
        speed = params['speed_gain'] * max(0.0, 1.0 - latency_ms / chunk_dt_ms)

    cost = (params['cost_spike'] * chunk_total_spikes
            + params['cost_synapse'] * n_active_synapses
            + params['cost_neuron'] * n_grown_neurons)

    return task + speed - cost


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


def _make_poisson_inputs(neu, exc, rate, params):
    pois = []
    for i in exc:
        p = PoissonInput(target=neu[i], target_var='v', N=1, rate=rate,
                          weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)
    return pois


def run_plastic_experiment(path_comp, path_con, neu_exc, neu_reward,
                            params=plastic_params, n_chunks=10, review_every=5,
                            r_poi=None, verbose=True):
    '''Run a reward/growth/neurogenesis pilot on a connectome.

    Neurons in `neu_exc` receive Poisson input for the whole run (the
    "sensory" side). Every `chunk_dt`, we check whether any neuron in
    `neu_reward` (the "motor"/output side) spiked since the last check; that
    gates a dopamine-style weight update via `update_weights`. Every
    `review_every` chunks, `apply_growth_atrophy` runs a structural pass
    (ceiling growth / atrophy), and if `params['neurogenesis']` is set, any
    neuron that has been pinned at its ceiling for `sat_patience` reviews in a
    row is duplicated: the live network is snapshotted, rebuilt one neuron
    larger per candidate via `rebuild_model_with_growth`, and the run
    continues on the bigger network.

    Parameters
    ----------
    neu_exc : list of flywire IDs
        neurons driven with Poisson input (stand-in for a sensory cue)
    neu_reward : list of flywire IDs
        neurons whose firing defines the reward signal (stand-in for a body-part/motor readout)
    n_chunks : int
        number of `params['chunk_dt']`-long chunks to run
    review_every : int
        run `apply_growth_atrophy` (and the neurogenesis check) every this many chunks
    r_poi : brian2 Hz quantity, optional
        override for the Poisson input rate (defaults to `params['r_poi']`)

    Returns
    -------
    neu, syn, spk_mon : brian2 objects (final state, possibly grown)
    log : list of dict
        one entry per chunk with reward/weight/eligibility/growth/neurogenesis stats
    grown : list of dict
        one entry per neuron actually grown: {'index', 'parent', 'chunk'}
    '''

    neu, syn, spk_mon, df_comp = create_plastic_model(path_comp, path_con, params)
    n_original = len(df_comp)

    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    exc = [flyid2i[n] for n in neu_exc]
    reward_idx = [flyid2i[n] for n in neu_reward]

    rate = r_poi if r_poi is not None else params['r_poi']
    pois = _make_poisson_inputs(neu, exc, rate, params)
    net = Network(neu, syn, spk_mon, *pois)

    sat_streak = np.zeros(n_original + params['max_new_neurons'])
    grown = []

    log = []
    prev_reward_spikes = 0
    prev_total_spikes = 0
    for c in range(n_chunks):
        chunk_start_ms = float(net.t / ms)
        net.run(params['chunk_dt'])

        counts = np.asarray(spk_mon.count[:])
        total_reward_spikes = int(counts[reward_idx].sum())
        chunk_reward_spikes = total_reward_spikes - prev_reward_spikes
        prev_reward_spikes = total_reward_spikes

        total_spikes_cum = int(counts.sum())
        chunk_total_spikes = total_spikes_cum - prev_total_spikes
        prev_total_spikes = total_spikes_cum

        latency_ms = None
        if chunk_reward_spikes > 0:
            latency_ms = _reward_latency_ms(spk_mon, reward_idx, chunk_start_ms)

        n_active_synapses = int(np.count_nonzero(
            np.abs(np.asarray(syn.w[:] / mV)) >= (params['active_syn_thr'] / mV)))
        n_grown_neurons = len(grown)

        dopamine = compute_efficiency_reward(
            chunk_reward_spikes, latency_ms, chunk_total_spikes,
            n_active_synapses, n_grown_neurons, params)
        w_new, elig, wmax = update_weights(syn, dopamine, params)

        entry = {
            'chunk': c,
            'n_neurons': len(neu),
            'reward_spikes': chunk_reward_spikes,
            'latency_ms': latency_ms,
            'dopamine': dopamine,
            'chunk_total_spikes': chunk_total_spikes,
            'n_active_synapses': n_active_synapses,
            'mean_abs_w_mV': float(np.mean(np.abs(w_new))),
            'mean_abs_elig_mV': float(np.mean(np.abs(elig))),
            'mean_w_max_mV': float(np.mean(wmax)),
        }

        if (c + 1) % review_every == 0:
            # judge neurogenesis against the ceiling as it stood BEFORE this
            # review's growth pass raises it -- otherwise a synapse that just
            # got more headroom never looks "stuck" long enough to trigger it
            candidates = []
            if params.get('neurogenesis', False) and len(grown) < params['max_new_neurons']:
                pre_state = snapshot_state(neu, syn)
                candidates = find_growth_candidates(pre_state, sat_streak[:len(neu)], params)
                candidates = candidates[: params['max_new_neurons'] - len(grown)]

            frac_atrophied, frac_grown = apply_growth_atrophy(syn, params)
            entry['frac_atrophied'] = frac_atrophied
            entry['frac_grown'] = frac_grown

            if candidates:
                state = snapshot_state(neu, syn)
                neu, syn, spk_mon, new_indices = rebuild_model_with_growth(state, params, candidates)
                prev_reward_spikes = 0  # fresh SpikeMonitor
                prev_total_spikes = 0
                pois = _make_poisson_inputs(neu, exc, rate, params)
                net = Network(neu, syn, spk_mon, *pois)

                for parent, new_idx in zip(candidates, new_indices):
                    grown.append({'index': new_idx, 'parent': parent, 'chunk': c})

                    sat_streak = np.concatenate([sat_streak, np.zeros(len(new_indices))])
                    entry['neurons_grown'] = new_indices
                    entry['n_neurons'] = len(neu)

        log.append(entry)
        if verbose:
            print(entry)

    return neu, syn, spk_mon, log, grown
