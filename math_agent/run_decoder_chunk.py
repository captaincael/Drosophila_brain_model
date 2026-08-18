'''Closed-loop sensory-encode / motor-decode experiment on the partial
connectome -- no external AI agent involved. The connectome itself attempts
the task:

- SENSORY (information intake): the 19 gustatory seed neurons are split into
  8 groups, one per digit of the two 4-digit operands. Each group's Poisson
  input rate is set proportional to that digit's value (0-9) -- so the
  operands are rate-coded directly into the sensory population instead of a
  flat, content-blind drive.
- MOTOR (the "decision" layer): 10 of the subgraph's 15 motor neurons are
  assigned to digit slots 0-9. After one chunk, whichever motor neuron fired
  the most is the network's predicted last digit of the product.
- REWARD: correct/incorrect prediction (of (a*b) % 10) drives dopamine,
  exactly as in run_math_chunk.py, but now class-differentiated
  (`update_weights_classed`): synapses feeding a motor neuron get a bigger
  reward-driven update ("motor neurons do the thinking" -- fast adaptation),
  synapses between two sensory neurons get a much smaller one (their job is
  to keep encoding information faithfully, not to chase reward).

Ground truth is computed directly (no LLM needed to grade single-digit
arithmetic), so this can run many trials quickly in a tight loop.
'''

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brian2 import ms, mV, Hz, Network, Synapses

from plasticity import (
    create_plastic_model, save_state, load_state, build_from_state,
    snapshot_state, apply_growth_atrophy, find_growth_candidates,
    rebuild_model_with_growth, update_weights_classed,
    plastic_params, _make_poisson_inputs_variable,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_COMP = os.path.join(_HERE, 'subgraph_comp_1hop_783.csv')
DEFAULT_CON = os.path.join(_HERE, 'subgraph_con_1hop_783.parquet')
DEFAULT_LOG_DIR = r'C:\Users\caele\OneDrive\Desktop\Project\Drosophila_brain_model\Test Logs'
ANNOTATIONS_PATH = os.path.join(_REPO_ROOT, 'annotations', 'flywire_783_neuron_annotations.tsv')
FLYWIRE_MATERIALIZATION = '783'

# same 19 confirmed-present gustatory (LB3) sensory seeds used in run_math_chunk.py
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
]

# all 15 motor (brain_motor_neuron) neurons found in the 783 subgraph; we use 10 of them
# as digit-slots 0-9 for the readout
MOTOR_POOL = [
    720575940607193986, 720575940610679876, 720575940614734754, 720575940618238523,
    720575940623415849, 720575940626094350, 720575940627410451, 720575940628826128,
    720575940628997123, 720575940629561347, 720575940629810748, 720575940644726432,
    720575940645528430, 720575940647474979, 720575940660219265,
]
MOTOR_DIGIT_SLOTS = sorted(MOTOR_POOL)[:10]

# 19 sensory neurons split across 8 digit positions (a3 a2 a1 a0 b3 b2 b1 b0)
SENSORY_GROUP_SIZES = [3, 3, 3, 2, 2, 2, 2, 2]
DIGIT_POSITIONS = ['a3', 'a2', 'a1', 'a0', 'b3', 'b2', 'b1', 'b0']


def build_params():
    params = dict(plastic_params)
    params['chunk_dt'] = 200 * ms     # longer integration window: more time for motor-layer spikes to accumulate/correlate
    params['growth_mult'] = 1.3
    params['w_max_floor'] = 0.2 * mV
    params['sat_frac_thr'] = 0.5
    params['sat_patience'] = 2
    params['lr'] = 0.2
    params['max_new_neurons'] = 10
    params['penalty'] = -1.0
    params['margin_gain'] = 0.6       # bonus for a decisive (not close) correct vote
    params['r_base'] = 40 * Hz        # sensory rate at digit value 0 (was 20)
    params['r_step'] = 30 * Hz        # additional rate per digit value 0-9 -> 40-310 Hz (was 20-155)
    params['sensory_mult'] = 0.2      # reward-weight multiplier: sensory<->sensory synapses (stay a stable encoder)
    params['motor_mult'] = 2.0        # reward-weight multiplier: synapses landing on a motor neuron ("does the thinking")
    params['w_lat_inh'] = -6 * mV     # fixed (non-plastic) lateral inhibition strength among the 10 motor digit-slots
    params['w_floor'] = 0.05 * mV     # homeostatic floor: excitatory synapses can never be punished all the way to 0
    params['motor_drive_boost'] = 3.0   # one-time startup boost to the raw connectome weight of any synapse landing on a motor neuron
    params['mutate_patience'] = 3       # consecutive wrong-and-won trials before a motor neuron's traits get mutated
    params['mutate_dampen_scale'] = 0.15  # non-sensory incoming synapses (the "free ride" pathway) scaled down by this
    params['mutate_sensory_jitter_mV'] = 1.5  # sensory-incoming synapses get reshuffled to a fresh random value in [0, this]
    return params


def build_lateral_inhibition(neu, motor_indices, params):
    '''Fixed, non-plastic all-to-all inhibitory wiring among the motor
    digit-slot neurons, so a spike in one immediately suppresses the others
    instead of one neuron structurally dominating every vote regardless of
    input. Deterministic from `motor_indices` alone -- rebuilt fresh every
    invocation, no state to persist.
    '''
    inh = Synapses(neu, neu, on_pre='g += w_lat_inh', delay=params['t_dly'],
                    namespace=params, name='lateral_inhibition')
    pre, post = [], []
    for i in motor_indices:
        for j in motor_indices:
            if i != j:
                pre.append(i)
                post.append(j)
    inh.connect(i=pre, j=post)
    return inh


def file_hash(path, n=8):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:n]


def connectome_id(comp_path, con_path):
    return 'flywire{}_1hop_sugar_mn9_comp-{}_con-{}'.format(
        FLYWIRE_MATERIALIZATION, file_hash(comp_path), file_hash(con_path))


def load_annotations():
    import pandas as pd
    df = pd.read_csv(ANNOTATIONS_PATH, sep='\t',
                      usecols=['root_id', 'super_class', 'cell_class', 'cell_type', 'side'],
                      low_memory=False)
    out = {}
    for row in df.itertuples(index=False):
        out[str(row.root_id)] = {
            'super_class': None if pd.isna(row.super_class) else row.super_class,
            'cell_class': None if pd.isna(row.cell_class) else row.cell_class,
            'cell_type': None if pd.isna(row.cell_type) else row.cell_type,
            'side': None if pd.isna(row.side) else row.side,
        }
    return out


def label_for(idx, i2flyid, grown_by_index):
    if idx in i2flyid:
        return i2flyid[idx]
    parent = grown_by_index.get(idx)
    if parent is None:
        return 'unknown:{}'.format(idx)
    return 'grown:{}:parent={}'.format(idx, label_for(parent, i2flyid, grown_by_index))


def annotate(flyid, annotations):
    if flyid.startswith('grown:') or flyid.startswith('unknown:'):
        return {'super_class': 'grown', 'cell_class': None, 'cell_type': None, 'side': None}
    return annotations.get(flyid, {'super_class': 'unmatched', 'cell_class': None,
                                    'cell_type': None, 'side': None})


def digit_groups():
    groups = []
    idx = 0
    for size in SENSORY_GROUP_SIZES:
        groups.append(NEU_SUGAR[idx:idx + size])
        idx += size
    return dict(zip(DIGIT_POSITIONS, groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--comp', default=DEFAULT_COMP)
    ap.add_argument('--con', default=DEFAULT_CON)
    ap.add_argument('--review-every', type=int, default=3)
    ap.add_argument('--log-dir', default=DEFAULT_LOG_DIR)
    ap.add_argument('--set-index', type=int, default=0)
    ap.add_argument('--trial-index', type=int, default=0)
    ap.add_argument('--a', type=int, default=None, help='operand A (1000-9999); random if omitted')
    ap.add_argument('--b', type=int, default=None, help='operand B (1000-9999); random if omitted')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    params = build_params()
    conn_id = connectome_id(args.comp, args.con)
    annotations = load_annotations()

    if os.path.exists(args.state):
        state = load_state(args.state)
        with open(args.meta) as f:
            meta = json.load(f)
        neu, syn, spk_mon = build_from_state(state, params)
        n_original = meta['n_original']
        grown = meta['grown']
        chunk_index = meta['chunk_index']
        flyid2i = meta['flyid2i']
        sat_streak = state['sat_streak'].copy()
        true_digit_hist = meta.get('true_digit_hist', [0] * 10)
        wrong_streak = meta.get('wrong_streak', [0] * 10)
    else:
        neu, syn, spk_mon, df_comp = create_plastic_model(args.comp, args.con, params)
        n_original = len(df_comp)
        grown = []
        chunk_index = 0
        sat_streak = np.zeros(n_original + params['max_new_neurons'])
        flyid2i = {str(int(j)): int(i) for i, j in enumerate(df_comp.index)}
        true_digit_hist = [0] * 10
        wrong_streak = [0] * 10

        # one-time startup boost: strengthen every synapse landing on a motor
        # neuron, so the sensory signal has enough amplitude to actually reach
        # and discriminate at the motor layer instead of dying out en route
        motor_idx_boost = set(flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS if str(m) in flyid2i)
        syn_j0 = np.asarray(syn.j[:])
        boost_mask = np.isin(syn_j0, list(motor_idx_boost))
        w0 = np.asarray(syn.w[:] / mV)
        wmax0 = np.asarray(syn.w_max[:] / mV)
        w0[boost_mask] = w0[boost_mask] * params['motor_drive_boost']
        wmax0[boost_mask] = np.maximum(wmax0[boost_mask], np.abs(w0[boost_mask]) * 1.05)
        syn.w = w0 * mV
        syn.w_max = wmax0 * mV

    i2flyid = {i: fid for fid, i in flyid2i.items()}
    grown_by_index = {g['index']: g['parent'] for g in grown}
    motor_indices = [flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS]
    lateral_inh = build_lateral_inhibition(neu, motor_indices, params)

    rng = np.random.default_rng(args.seed)
    a = args.a if args.a is not None else int(rng.integers(1000, 10000))
    b = args.b if args.b is not None else int(rng.integers(1000, 10000))
    true_digit = (a * b) % 10

    # majority-class baseline, computed from every true digit seen BEFORE this trial
    # (so it can't leak this trial's answer into its own reward)
    n_seen = sum(true_digit_hist)
    if n_seen == 0:
        majority_class, baseline_rate = None, 0.1  # uniform prior, no data yet
    else:
        majority_class = int(np.argmax(true_digit_hist))
        baseline_rate = true_digit_hist[majority_class] / n_seen

    # --- SENSORY: rate-code the operands into the 19 gustatory neurons ---
    groups = digit_groups()
    digit_values = [int(d) for d in '{:04d}'.format(a)] + [int(d) for d in '{:04d}'.format(b)]
    rate_by_index = {}
    encoding_log = {}
    for pos, d in zip(DIGIT_POSITIONS, digit_values):
        rate = params['r_base'] + d * params['r_step']
        ids = groups[pos]
        encoding_log[pos] = {'digit': d, 'rate_hz': float(rate / Hz), 'neuron_ids': ids}
        for fid in ids:
            rate_by_index[flyid2i[str(fid)]] = rate

    pois = _make_poisson_inputs_variable(neu, rate_by_index, params)
    net = Network(neu, syn, lateral_inh, spk_mon, *pois)
    net.run(params['chunk_dt'])

    counts = np.asarray(spk_mon.count[:])

    # --- MOTOR: decode the predicted digit (competing via lateral_inh) ---
    motor_counts = counts[motor_indices]
    predicted_digit = int(np.argmax(motor_counts))
    correct = predicted_digit == true_digit

    chunk_total_spikes = int(counts.sum())
    w_mV = np.asarray(syn.w[:] / mV)
    elig_mV = np.asarray(syn.elig[:] / mV)
    n_active_synapses = int(np.count_nonzero(np.abs(w_mV) >= (params['active_syn_thr'] / mV)))
    n_grown_neurons = len(grown)

    sorted_counts = sorted(motor_counts, reverse=True)
    top = int(sorted_counts[0])
    second = int(sorted_counts[1]) if len(sorted_counts) > 1 else 0
    margin = params['margin_gain'] * ((top - second) / top) if (correct and top > 0) else 0.0

    # baseline-normalized task reward: a correct guess that just matches the
    # majority class (what you'd get by ignoring the input and always
    # guessing the most common answer) is discounted; a correct guess that
    # beats the majority class (i.e. picked a minority digit AND got it
    # right, which requires actually using the sensory encoding) is boosted.
    # Being wrong is punished the same regardless -- this only changes how
    # much a *correct* answer is worth.
    if not correct:
        task = params['penalty']
    elif majority_class is not None and predicted_digit == majority_class:
        task = 1.0 * (1 - baseline_rate)
    else:
        task = 1.0 * (1 + baseline_rate)

    cost = (params['cost_spike'] * chunk_total_spikes
            + params['cost_synapse'] * n_active_synapses
            + params['cost_neuron'] * n_grown_neurons)
    dopamine = task + margin - cost

    # --- class-differentiated plasticity ---
    neuron_class = np.full(len(neu), 'other', dtype=object)
    sensory_idx = [flyid2i[str(s)] for s in NEU_SUGAR if str(s) in flyid2i]
    for idx in sensory_idx:
        neuron_class[idx] = 'sensory'
    for idx in motor_indices:
        neuron_class[idx] = 'motor'
    for g in grown:
        if g['index'] < len(neuron_class) and g['parent'] < len(neuron_class):
            neuron_class[g['index']] = neuron_class[g['parent']]

    syn_i = np.asarray(syn.i[:])
    syn_j = np.asarray(syn.j[:])
    pre_class = neuron_class[syn_i]
    post_class = neuron_class[syn_j]
    class_mult = np.ones(len(syn_i))
    class_mult[post_class == 'motor'] = params['motor_mult']
    class_mult[(pre_class == 'sensory') & (post_class == 'sensory')] = params['sensory_mult']

    w_new, elig, wmax = update_weights_classed(syn, dopamine, class_mult, params)

    # --- mutegen: a motor neuron that keeps WINNING the vote while being
    # WRONG, trial after trial, has some trait letting it fire regardless of
    # input -- since nothing dies here, the fix isn't to suppress it (that
    # throws away a neuron that's easy to excite, which is a useful trait)
    # but to MUTATE it: dampen whatever non-sensory ("free ride") input is
    # driving it unconditionally, and randomly reshuffle its sensory-driven
    # input so it's forced to try a different relationship to the actual
    # encoded digits. Real trial-and-error variation, not punishment -- the
    # normal reward loop is what selects whether the mutation was any good.
    if correct:
        wrong_streak[predicted_digit] = 0
        mutated_digit = None
    else:
        wrong_streak[predicted_digit] += 1
        mutated_digit = None
        if wrong_streak[predicted_digit] >= params['mutate_patience']:
            mutated_idx = motor_indices[predicted_digit]
            mask = syn_j == mutated_idx
            from_sensory = mask & (pre_class == 'sensory')
            from_other = mask & ~from_sensory

            w_new = w_new.copy()
            sign_local = np.asarray(syn.w_sign[:])
            w_new[from_other] = w_new[from_other] * params['mutate_dampen_scale']
            n_sens = int(from_sensory.sum())
            if n_sens > 0:
                jitter = rng.uniform(0.0, params['mutate_sensory_jitter_mV'], size=n_sens)
                w_new[from_sensory] = jitter * sign_local[from_sensory]
            syn.w = w_new * mV

            wrong_streak[predicted_digit] = 0
            mutated_digit = predicted_digit

    n_original_matched = sum(1 for fid in flyid2i if fid in annotations)
    connectome_state = {
        'flywire_materialization': FLYWIRE_MATERIALIZATION,
        'annotation_source': 'flyconnectome/flywire_annotations (main, 783)',
        'annotation_table_hash': file_hash(ANNOTATIONS_PATH),
        'n_neurons_original': n_original,
        'n_neurons_current': len(neu),
        'n_neurons_grown_total': n_grown_neurons,
        'n_synapses_total': len(syn),
        'n_active_synapses': n_active_synapses,
        'annotation_coverage_original': n_original_matched / n_original if n_original else None,
    }

    motor_slots_log = [
        {
            'digit_slot': k,
            'flyid': MOTOR_DIGIT_SLOTS[k],
            'cell_type': annotate(str(MOTOR_DIGIT_SLOTS[k]), annotations)['cell_type'],
            'spikes': int(motor_counts[k]),
        }
        for k in range(len(MOTOR_DIGIT_SLOTS))
    ]

    counts_arr = counts
    spiking_neurons = []
    for idx, c in enumerate(counts_arr):
        if c <= 0:
            continue
        flyid = label_for(int(idx), i2flyid, grown_by_index)
        entry = {'id': flyid, 'spikes': int(c), 'role': str(neuron_class[idx])}
        entry.update(annotate(flyid, annotations))
        spiking_neurons.append(entry)
    spiking_neurons.sort(key=lambda d: -d['spikes'])

    top_k = min(15, len(elig_mV))
    top_idx = np.argsort(-np.abs(elig_mV))[:top_k]
    top_synapses = []
    for k in top_idx:
        if abs(elig_mV[k]) <= 0:
            continue
        pre_id = label_for(int(syn_i[k]), i2flyid, grown_by_index)
        post_id = label_for(int(syn_j[k]), i2flyid, grown_by_index)
        top_synapses.append({
            'pre': pre_id, 'pre_role': str(neuron_class[syn_i[k]]),
            'post': post_id, 'post_role': str(neuron_class[syn_j[k]]),
            'w_mV': float(w_mV[k]), 'elig_mV': float(elig_mV[k]),
            'class_mult_applied': float(class_mult[k]),
        })

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'connectome_id': conn_id,
        'connectome_state': connectome_state,
        'set_index': args.set_index,
        'trial_index': args.trial_index,
        'chunk_index': chunk_index,
        'operand_a': a,
        'operand_b': b,
        'true_last_digit': true_digit,
        'predicted_last_digit': predicted_digit,
        'correct': correct,
        'majority_class_baseline': {'majority_class': majority_class, 'baseline_rate': baseline_rate,
                                     'n_trials_seen': n_seen, 'beat_baseline': bool(correct and predicted_digit != majority_class)},
        'mutegen': {'wrong_streak': list(wrong_streak), 'mutated_digit_slot': mutated_digit,
                    'mutated_flyid': MOTOR_DIGIT_SLOTS[mutated_digit] if mutated_digit is not None else None},
        'sensory_encoding': encoding_log,
        'motor_decoding': motor_slots_log,
        'vote_margin': top - second,
        'dopamine': dopamine,
        'dopamine_task': task,
        'dopamine_margin_bonus': margin,
        'dopamine_cost': cost,
        'n_neurons': len(neu),
        'chunk_total_spikes': chunk_total_spikes,
        'n_active_synapses': n_active_synapses,
        'mean_abs_w_mV': float(np.mean(np.abs(w_new))),
        'mean_abs_elig_mV': float(np.mean(np.abs(elig))),
        'mean_w_max_mV': float(np.mean(wmax)),
        'spiking_neurons': spiking_neurons,
        'n_spiking_neurons': len(spiking_neurons),
        'top_synapses_by_eligibility': top_synapses,
        'neurons_grown': [],
    }

    if (chunk_index + 1) % args.review_every == 0:
        candidates = []
        if params.get('neurogenesis', False) and len(grown) < params['max_new_neurons']:
            pre_state = snapshot_state(neu, syn)
            candidates = find_growth_candidates(pre_state, sat_streak[:len(neu)], params)
            candidates = candidates[: params['max_new_neurons'] - len(grown)]

        frac_atrophied, frac_grown = apply_growth_atrophy(syn, params)
        result['frac_atrophied'] = frac_atrophied
        result['frac_grown'] = frac_grown

        if candidates:
            state = snapshot_state(neu, syn)
            neu, syn, spk_mon, new_indices = rebuild_model_with_growth(state, params, candidates)
            for parent, new_idx in zip(candidates, new_indices):
                grown.append({'index': int(new_idx), 'parent': int(parent), 'chunk': chunk_index})
            sat_streak = np.concatenate([sat_streak, np.zeros(len(new_indices))])
            result['neurons_grown'] = [
                label_for(int(idx), i2flyid, {g['index']: g['parent'] for g in grown})
                for idx in new_indices
            ]
            result['n_neurons'] = len(neu)

    final_state = snapshot_state(neu, syn)
    final_state['sat_streak'] = sat_streak[:len(neu)]
    save_state(final_state, args.state)

    true_digit_hist[true_digit] += 1
    meta_out = {'flyid2i': flyid2i, 'n_original': n_original, 'grown': grown,
                'chunk_index': chunk_index + 1, 'true_digit_hist': true_digit_hist,
                'wrong_streak': list(wrong_streak)}
    with open(args.meta, 'w') as f:
        json.dump(meta_out, f)

    os.makedirs(args.log_dir, exist_ok=True)
    trial_log_path = os.path.join(
        args.log_dir, 'decoder_flywire{}_set{:02d}_trial{:03d}.json'.format(
            FLYWIRE_MATERIALIZATION, args.set_index, args.trial_index))
    with open(trial_log_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.log_dir, 'decoder_run_log.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(json.dumps({k: result[k] for k in result if k not in
                       ('spiking_neurons', 'top_synapses_by_eligibility', 'sensory_encoding')}))


if __name__ == '__main__':
    main()
