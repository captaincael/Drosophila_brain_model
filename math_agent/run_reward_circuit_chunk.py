'''Replaces the hand-coded dopamine formula with an actual dopaminergic
circuit built from real Central-class neurons, in real, verified synaptic
contact with a teaching-signal pathway. Deliberately kept separate from the
memory system (run_memory_chunk.py) so this mechanism's effect is cleanly
isolated and testable on its own.

Each trial now runs in two phases on the SAME live network (state carries
over between them automatically -- v, g, w, elig live on the NeuronGroup/
Synapses objects themselves, not on whatever Network wrapper calls run()):

1. DECODE phase (existing sensory-encode / lateral-inhibition motor-decode
   mechanics from run_decoder_chunk.py): the operand digits are rate-coded
   into the 19 gustatory sensory neurons, the network runs, and whichever of
   10 motor neurons fires most is the predicted digit.

2. FEEDBACK phase: correctness is graded externally (still requires knowing
   ground truth -- that part is unavoidable in any reward-driven system,
   biological or not) and used ONLY to pick which of two small teaching-
   signal neuron sets gets an external Poisson pulse: CORRECT_TEACHERS if
   right, INCORRECT_TEACHERS if wrong. These are real gustatory-adjacent
   Central neurons with no other role. The network runs a second, short
   chunk. Whichever dopaminergic pool -- REWARD_DA (real postsynaptic
   targets of CORRECT_TEACHERS) or PUNISH_DA (real postsynaptic targets of
   INCORRECT_TEACHERS) -- actually fires in response is what sets the
   dopamine value: `dopamine = gain * (reward_da_spikes - punish_da_spikes)`.
   This is a real neural response to a real stimulus, not a formula -- it
   also depends on whatever else the network is doing, not just the
   correct/incorrect signal alone.

That dopamine value is then applied exactly as before (update_weights_classed,
broad/network-wide, same as every other experiment this session) using the
eligibility trace built up during the decode phase, which is still warm when
the feedback phase runs immediately after.
'''

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brian2 import ms, mV, Hz, Network, Synapses, SpikeMonitor

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

NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
]
MOTOR_POOL = [
    720575940607193986, 720575940610679876, 720575940614734754, 720575940618238523,
    720575940623415849, 720575940626094350, 720575940627410451, 720575940628826128,
    720575940628997123, 720575940629561347, 720575940629810748, 720575940644726432,
    720575940645528430, 720575940647474979, 720575940660219265,
]
MOTOR_DIGIT_SLOTS = sorted(MOTOR_POOL)[:10]

# teaching signal: real Central neurons, no other role, driven externally after grading
CORRECT_TEACHERS = [720575940621256384, 720575940631340815, 720575940637216624]
INCORRECT_TEACHERS = [720575940623211725, 720575940619771375, 720575940631997032]

# dopaminergic pools: real, verified EXCITATORY postsynaptic targets of the teacher sets
# above (the original picks were "real targets" but several of the strongest edges were
# inhibitory -- CORRECT_TEACHERS->REWARD_DA was net-inhibitory, and INCORRECT_TEACHERS had
# zero excitatory outgoing edges in this subgraph at all, which is why the original
# REWARD_DA never fired once in 25 trials. These are re-picked filtering for Excitatory==1
# and ranked by summed connectivity strength (94-162 here, vs 1-16 for the old picks).
REWARD_DA = [720575940633941293, 720575940638604915, 720575940620359956, 720575940638103349]
PUNISH_DA = [720575940621261296, 720575940630820919, 720575940637578853, 720575940615410911]

SENSORY_GROUP_SIZES = [3, 3, 3, 2, 2, 2, 2, 2]
DIGIT_POSITIONS = ['a3', 'a2', 'a1', 'a0', 'b3', 'b2', 'b1', 'b0']


def build_params():
    params = dict(plastic_params)
    params['chunk_dt'] = 200 * ms
    params['feedback_dt'] = 100 * ms
    params['growth_mult'] = 1.3
    params['w_max_floor'] = 0.2 * mV
    params['sat_frac_thr'] = 0.5
    params['sat_patience'] = 2
    params['lr'] = 0.2
    params['max_new_neurons'] = 10
    params['r_base'] = 40 * Hz
    params['r_step'] = 30 * Hz
    params['teacher_rate'] = 200 * Hz
    params['sensory_mult'] = 0.2
    params['motor_mult'] = 2.0
    params['w_lat_inh'] = -6 * mV
    params['w_floor'] = 0.05 * mV
    params['motor_drive_boost'] = 3.0
    params['reward_circuit_boost'] = 3.0  # startup boost on teacher->DA synapses, same rationale as motor_drive_boost
    params['mutate_patience'] = 3
    params['mutate_dampen_scale'] = 0.15
    params['mutate_sensory_jitter_mV'] = 1.5
    params['cost_spike'] = 5e-5
    params['cost_synapse'] = 2e-6
    params['da_gain'] = 0.3  # dopamine per net (reward_da - punish_da) spike, averaged per DA neuron
    return params


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


def digit_groups():
    groups = []
    idx = 0
    for size in SENSORY_GROUP_SIZES:
        groups.append(NEU_SUGAR[idx:idx + size])
        idx += size
    return dict(zip(DIGIT_POSITIONS, groups))


def build_lateral_inhibition(neu, motor_indices, params):
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
    ap.add_argument('--a', type=int, default=None)
    ap.add_argument('--b', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    params = build_params()
    conn_id = connectome_id(args.comp, args.con)
    annotations = load_annotations()
    rng = np.random.default_rng(args.seed)

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
        wrong_streak = meta.get('wrong_streak', [0] * 10)
    else:
        neu, syn, spk_mon, df_comp = create_plastic_model(args.comp, args.con, params)
        n_original = len(df_comp)
        grown = []
        chunk_index = 0
        sat_streak = np.zeros(n_original + params['max_new_neurons'])
        flyid2i = {str(int(j)): int(i) for i, j in enumerate(df_comp.index)}
        wrong_streak = [0] * 10

        # startup boosts: motor-incoming (as before) and teacher->DA synapses
        # (same rationale: raw connectome weights are too weak for these small,
        # newly-repurposed pathways to reliably fire without help)
        motor_idx_boost = set(flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS if str(m) in flyid2i)
        correct_teacher_idx = set(flyid2i[str(m)] for m in CORRECT_TEACHERS if str(m) in flyid2i)
        incorrect_teacher_idx = set(flyid2i[str(m)] for m in INCORRECT_TEACHERS if str(m) in flyid2i)
        reward_da_idx = set(flyid2i[str(m)] for m in REWARD_DA if str(m) in flyid2i)
        punish_da_idx = set(flyid2i[str(m)] for m in PUNISH_DA if str(m) in flyid2i)

        syn_i0 = np.asarray(syn.i[:])
        syn_j0 = np.asarray(syn.j[:])
        motor_mask = np.isin(syn_j0, list(motor_idx_boost))
        da_mask = ((np.isin(syn_i0, list(correct_teacher_idx)) & np.isin(syn_j0, list(reward_da_idx)))
                   | (np.isin(syn_i0, list(incorrect_teacher_idx)) & np.isin(syn_j0, list(punish_da_idx))))

        w0 = np.asarray(syn.w[:] / mV)
        wmax0 = np.asarray(syn.w_max[:] / mV)
        w0[motor_mask] = w0[motor_mask] * params['motor_drive_boost']
        w0[da_mask] = w0[da_mask] * params['reward_circuit_boost']
        wmax0 = np.maximum(wmax0, np.abs(w0) * 1.05)
        syn.w = w0 * mV
        syn.w_max = wmax0 * mV

    i2flyid = {i: fid for fid, i in flyid2i.items()}
    grown_by_index = {g['index']: g['parent'] for g in grown}
    motor_indices = [flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS]
    correct_teacher_indices = [flyid2i[str(m)] for m in CORRECT_TEACHERS]
    incorrect_teacher_indices = [flyid2i[str(m)] for m in INCORRECT_TEACHERS]
    reward_da_indices = [flyid2i[str(m)] for m in REWARD_DA]
    punish_da_indices = [flyid2i[str(m)] for m in PUNISH_DA]
    lateral_inh = build_lateral_inhibition(neu, motor_indices, params)

    a = args.a if args.a is not None else int(rng.integers(1000, 10000))
    b = args.b if args.b is not None else int(rng.integers(1000, 10000))
    true_digit = (a * b) % 10

    # --- PHASE 1: decode ---
    groups = digit_groups()
    digit_values = [int(d) for d in '{:04d}'.format(a)] + [int(d) for d in '{:04d}'.format(b)]
    rate_by_index = {}
    for pos, d in zip(DIGIT_POSITIONS, digit_values):
        rate = params['r_base'] + d * params['r_step']
        for fid in groups[pos]:
            rate_by_index[flyid2i[str(fid)]] = rate

    pois1 = _make_poisson_inputs_variable(neu, rate_by_index, params)
    spk_mon = SpikeMonitor(neu)
    net = Network(neu, syn, lateral_inh, spk_mon)
    net.add(*pois1)
    net.run(params['chunk_dt'])
    counts1 = np.asarray(spk_mon.count[:]).copy()
    net.remove(*pois1)

    motor_counts = counts1[motor_indices]
    predicted_digit = int(np.argmax(motor_counts))
    correct = predicted_digit == true_digit

    # --- PHASE 2: feedback (real dopaminergic circuit), same live Network/state ---
    teacher_indices = correct_teacher_indices if correct else incorrect_teacher_indices
    teacher_rate_by_index = {idx: params['teacher_rate'] for idx in teacher_indices}
    pois2 = _make_poisson_inputs_variable(neu, teacher_rate_by_index, params)
    net.add(*pois2)
    net.run(params['feedback_dt'])
    counts_total = np.asarray(spk_mon.count[:]).copy()
    net.remove(*pois2)

    counts2 = counts_total - counts1  # phase-2-only spikes
    reward_da_spikes = int(counts2[reward_da_indices].sum())
    punish_da_spikes = int(counts2[punish_da_indices].sum())

    chunk_total_spikes = int(counts1.sum() + counts2.sum())
    w_mV = np.asarray(syn.w[:] / mV)
    elig_mV = np.asarray(syn.elig[:] / mV)
    n_active_synapses = int(np.count_nonzero(np.abs(w_mV) >= (params['active_syn_thr'] / mV)))
    n_grown_neurons = len(grown)

    cost = (params['cost_spike'] * chunk_total_spikes
            + params['cost_synapse'] * n_active_synapses
            + params['cost_neuron'] * n_grown_neurons)
    dopamine = params['da_gain'] * (reward_da_spikes - punish_da_spikes) / max(len(REWARD_DA), 1) - cost

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

    # --- mutegen (motor layer, unaffected by the new circuit) ---
    mutated_digit = None
    if correct:
        wrong_streak[predicted_digit] = 0
    else:
        wrong_streak[predicted_digit] += 1
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
        'n_neurons_original': n_original, 'n_neurons_current': len(neu),
        'n_neurons_grown_total': n_grown_neurons, 'n_synapses_total': len(syn),
        'n_active_synapses': n_active_synapses,
        'annotation_coverage_original': n_original_matched / n_original if n_original else None,
    }

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'connectome_id': conn_id, 'connectome_state': connectome_state,
        'set_index': args.set_index, 'trial_index': args.trial_index, 'chunk_index': chunk_index,
        'operand_a': a, 'operand_b': b, 'true_last_digit': true_digit,
        'predicted_last_digit': predicted_digit, 'correct': correct,
        'teacher_signal': 'correct' if correct else 'incorrect',
        'reward_da_spikes': reward_da_spikes, 'punish_da_spikes': punish_da_spikes,
        'dopamine': dopamine, 'mutated_digit_slot': mutated_digit,
        'n_neurons': len(neu), 'chunk_total_spikes': chunk_total_spikes,
        'n_active_synapses': n_active_synapses,
        'mean_abs_w_mV': float(np.mean(np.abs(w_new))),
        'mean_abs_elig_mV': float(np.mean(np.abs(elig))),
        'phase1_spikes': int(counts1.sum()), 'phase2_spikes': int(counts2.sum()),
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
            state_snap = snapshot_state(neu, syn)
            neu, syn, spk_mon, new_indices = rebuild_model_with_growth(state_snap, params, candidates)
            for parent, new_idx in zip(candidates, new_indices):
                grown.append({'index': int(new_idx), 'parent': int(parent), 'chunk': chunk_index})
            sat_streak = np.concatenate([sat_streak, np.zeros(len(new_indices))])
            result['neurons_grown'] = new_indices
            result['n_neurons'] = len(neu)

    final_state = snapshot_state(neu, syn)
    final_state['sat_streak'] = sat_streak[:len(neu)]
    save_state(final_state, args.state)

    meta_out = {'flyid2i': flyid2i, 'n_original': n_original, 'grown': grown,
                'chunk_index': chunk_index + 1, 'wrong_streak': wrong_streak}
    with open(args.meta, 'w') as f:
        json.dump(meta_out, f)

    os.makedirs(args.log_dir, exist_ok=True)
    trial_log_path = os.path.join(
        args.log_dir, 'rewardcircuit_flywire{}_set{:02d}_trial{:03d}.json'.format(
            FLYWIRE_MATERIALIZATION, args.set_index, args.trial_index))
    with open(trial_log_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.log_dir, 'rewardcircuit_run_log.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(json.dumps(result))


if __name__ == '__main__':
    main()
