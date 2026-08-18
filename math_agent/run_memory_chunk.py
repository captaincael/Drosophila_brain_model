'''Adds real associative memory to the sensory-encode/motor-decode decoder
experiment (run_decoder_chunk.py), living on top of a small set of Central-
class neurons.

- Memory_Short and Memory_Long neurons each hold a small buffer of "episode"
  embeddings (32-dim feature vectors describing a trial: the operand digits,
  and once known, the true/predicted digit and correctness).
- WRITE: any active memory neuron that actually fires during a trial writes
  that trial's full episode into its own buffer (FIFO once full).
- READ / retrieval: before a trial runs, its operand digits (only) are
  embedded into a query vector and compared (cosine similarity) against
  every memory neuron's stored episodes. A neuron whose best match clears
  `similarity_threshold` counts as "referenced" this trial -- content-
  addressable recall, not just "did it spike."
- REWARD: a referenced neuron gets a similarity-scaled dopamine bonus
  applied only to its own incoming synapses (separate from, and on top of,
  the shared task-correctness dopamine everything else responds to).
- Memory_Short has a hard ~100KB storage budget (few hundred episodes).
  A Short neuron that goes `short_atrophy_patience` trials with no
  reference undergoes a real neurogenesis event: a new Memory_Long neuron
  is grown (`rebuild_model_with_growth`, same mechanism as the decoder's
  mutegen/neurogenesis), inherits its episode buffer, and the old Short
  neuron is retired ("graduated" -- no longer written to or checked).
- Memory_Long atrophies far more slowly (much longer patience) but, if it
  goes `long_die_patience` trials with truly zero references, is actually
  removed from the network (`prune_neurons` -- the first real neuron death
  in this codebase; everything before this only ever grew).
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
    rebuild_model_with_growth, prune_neurons, update_weights_classed,
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

# top-8 Central-class neurons in the subgraph by degree, seeded as Memory_Short
MEMORY_SHORT_SEEDS = [
    720575940625867056, 720575940631349335, 720575940610482883, 720575940614660519,
    720575940617787963, 720575940607272649, 720575940617678566, 720575940627383685,
]

SENSORY_GROUP_SIZES = [3, 3, 3, 2, 2, 2, 2, 2]
DIGIT_POSITIONS = ['a3', 'a2', 'a1', 'a0', 'b3', 'b2', 'b1', 'b0']

EMB_DIM = 32
ENTRY_OVERHEAD_BYTES = 16  # rough per-entry bookkeeping (trial index, flags) beyond the raw vector
# dims used for similarity matching: digit/situation features only (0-7, 11-14).
# Outcome fields (8-10: true/predicted digit, correct) are stored for content but
# excluded from matching -- recall happens before the outcome is known, so a
# query (outcome masked to -1) must never be compared against a stored vector's
# filled-in outcome, or even an identical situation would score a low similarity.
SITUATION_DIMS = list(range(0, 8)) + list(range(11, 15))


def build_params():
    params = dict(plastic_params)
    params['chunk_dt'] = 200 * ms
    params['growth_mult'] = 1.3
    params['w_max_floor'] = 0.2 * mV
    params['sat_frac_thr'] = 0.5
    params['sat_patience'] = 2
    params['lr'] = 0.2
    params['max_new_neurons'] = 10
    params['penalty'] = -1.0
    params['margin_gain'] = 0.6
    params['r_base'] = 40 * Hz
    params['r_step'] = 30 * Hz
    params['sensory_mult'] = 0.2
    params['motor_mult'] = 2.0
    params['memory_mult'] = 1.5       # reward-weight multiplier: synapses landing on an active memory neuron
    params['w_lat_inh'] = -6 * mV
    params['w_floor'] = 0.05 * mV
    params['motor_drive_boost'] = 3.0
    params['mutate_patience'] = 3
    params['mutate_dampen_scale'] = 0.15
    params['mutate_sensory_jitter_mV'] = 1.5
    params['cost_spike'] = 5e-5
    params['cost_synapse'] = 2e-6

    # memory-specific
    params['short_capacity_bytes'] = 100 * 1024
    params['long_capacity_entries'] = 100000  # effectively unlimited at our trial counts
    params['similarity_threshold'] = 0.85
    params['memory_dopamine_gain'] = 1.2
    params['recall_rate_gain'] = 150 * Hz  # extra Poisson drive onto the recalled motor slot, scaled by similarity (0-1)
    params['short_atrophy_patience'] = 5    # trials with no reference before Short -> Long transform
    params['long_die_patience'] = 25        # trials with no reference before a Long neuron is pruned ("extremely slowly")
    return params


def short_capacity_entries(params):
    return max(1, params['short_capacity_bytes'] // (EMB_DIM * 4 + ENTRY_OVERHEAD_BYTES))


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


def episode_embedding(a, b, true_digit=None, predicted_digit=None, correct=None):
    '''32-dim feature-vector embedding of a trial "episode." Used both as the
    write vector (outcome fields filled in) and, with outcome fields left
    unknown, as the retrieval query (situation only, before the outcome is
    known -- real recall is proactive, not hindsight).'''
    digits = [int(d) for d in '{:04d}'.format(a)] + [int(d) for d in '{:04d}'.format(b)]
    v = np.zeros(EMB_DIM)
    v[0:8] = [d / 9.0 for d in digits]
    v[8] = (true_digit / 9.0) if true_digit is not None else -1.0
    v[9] = (predicted_digit / 9.0) if predicted_digit is not None else -1.0
    v[10] = -1.0 if correct is None else (1.0 if correct else 0.0)
    v[11] = sum(digits[0:4]) / 36.0
    v[12] = sum(digits[4:8]) / 36.0
    v[13] = a / 9999.0
    v[14] = b / 9999.0
    return v


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def situation_sim(query_vec, stored_vec):
    '''Cosine similarity restricted to the situation dims -- see SITUATION_DIMS.'''
    q = np.asarray(query_vec)[SITUATION_DIMS]
    s = np.asarray(stored_vec)[SITUATION_DIMS]
    return cosine_sim(q, s)


def remap_indices(old_to_new, flyid2i, grown, memory_index_map):
    '''Apply a prune_neurons remap to every index-keyed structure we persist.'''
    new_flyid2i = {}
    for fid, idx in flyid2i.items():
        new_idx = old_to_new.get(idx)
        if new_idx is not None:
            new_flyid2i[fid] = new_idx

    new_grown = []
    for g in grown:
        new_index = old_to_new.get(g['index'])
        new_parent = old_to_new.get(g['parent'])
        if new_index is not None:
            new_grown.append({'index': new_index, 'parent': (new_parent if new_parent is not None else -1),
                               'chunk': g['chunk']})

    new_memory_index_map = {}
    for key, idx in memory_index_map.items():
        new_idx = old_to_new.get(idx)
        if new_idx is not None:
            new_memory_index_map[key] = new_idx

    return new_flyid2i, new_grown, new_memory_index_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--memory-bank', required=True)
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
        true_digit_hist = meta.get('true_digit_hist', [0] * 10)
        wrong_streak = meta.get('wrong_streak', [0] * 10)
        memory_index_map = meta.get('memory_index_map', {})
        mem_long_counter = meta.get('mem_long_counter', 0)
        with open(args.memory_bank) as f:
            memory_bank = json.load(f)
    else:
        neu, syn, spk_mon, df_comp = create_plastic_model(args.comp, args.con, params)
        n_original = len(df_comp)
        grown = []
        chunk_index = 0
        sat_streak = np.zeros(n_original + params['max_new_neurons'])
        flyid2i = {str(int(j)): int(i) for i, j in enumerate(df_comp.index)}
        true_digit_hist = [0] * 10
        wrong_streak = [0] * 10
        mem_long_counter = 0

        motor_idx_boost = set(flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS if str(m) in flyid2i)
        syn_j0 = np.asarray(syn.j[:])
        boost_mask = np.isin(syn_j0, list(motor_idx_boost))
        w0 = np.asarray(syn.w[:] / mV)
        wmax0 = np.asarray(syn.w_max[:] / mV)
        w0[boost_mask] = w0[boost_mask] * params['motor_drive_boost']
        wmax0[boost_mask] = np.maximum(wmax0[boost_mask], np.abs(w0[boost_mask]) * 1.05)
        syn.w = w0 * mV
        syn.w_max = wmax0 * mV

        memory_index_map = {str(m): flyid2i[str(m)] for m in MEMORY_SHORT_SEEDS if str(m) in flyid2i}
        memory_bank = {
            key: {'role': 'short', 'embeddings': [], 'meta': [], 'created_trial': 0,
                  'no_reference_streak': 0, 'last_referenced_trial': None}
            for key in memory_index_map
        }

    i2flyid = {i: fid for fid, i in flyid2i.items()}
    grown_by_index = {g['index']: g['parent'] for g in grown}
    motor_indices = [flyid2i[str(m)] for m in MOTOR_DIGIT_SLOTS]
    lateral_inh = build_lateral_inhibition(neu, motor_indices, params)

    a = args.a if args.a is not None else int(rng.integers(1000, 10000))
    b = args.b if args.b is not None else int(rng.integers(1000, 10000))
    true_digit = (a * b) % 10

    n_seen = sum(true_digit_hist)
    if n_seen == 0:
        majority_class, baseline_rate = None, 0.1
    else:
        majority_class = int(np.argmax(true_digit_hist))
        baseline_rate = true_digit_hist[majority_class] / n_seen

    # --- MEMORY: retrieval (before the trial runs -- situation cues only) ---
    active_memory = {k: idx for k, idx in memory_index_map.items()
                      if memory_bank.get(k, {}).get('role') in ('short', 'long')}
    query_vec = episode_embedding(a, b)
    referenced = {}
    best_similarity = {}
    recalled_digit = {}
    for key, idx in active_memory.items():
        embs = memory_bank[key]['embeddings']
        metas = memory_bank[key]['meta']
        sims = [situation_sim(query_vec, e) for e in embs]
        if sims:
            best_i = int(np.argmax(sims))
            best_similarity[key] = sims[best_i]
            recalled_digit[key] = metas[best_i]['true_digit']
        else:
            best_similarity[key] = 0.0
            recalled_digit[key] = None
        referenced[key] = best_similarity[key] >= params['similarity_threshold']

    # --- CAUSAL RECALL: a referenced memory injects extra excitatory drive
    # directly onto the motor slot for the digit it remembers, biasing this
    # trial's vote toward that recollection -- competing with (or reinforcing)
    # the sensory-driven signal instead of just sitting in a separate reward
    # ledger. Multiple referenced memories pointing at the same digit stack.
    recall_rate_by_slot = {}
    recall_events = []
    for key in active_memory:
        if not referenced[key] or recalled_digit[key] is None:
            continue
        digit = recalled_digit[key]
        slot_idx = motor_indices[digit]
        boost = params['recall_rate_gain'] * best_similarity[key]
        recall_rate_by_slot[slot_idx] = recall_rate_by_slot.get(slot_idx, 0 * Hz) + boost
        recall_events.append({'key': key, 'recalled_digit': digit,
                               'similarity': best_similarity[key], 'boost_hz': float(boost / Hz)})

    # --- SENSORY: rate-code the operands ---
    groups = digit_groups()
    digit_values = [int(d) for d in '{:04d}'.format(a)] + [int(d) for d in '{:04d}'.format(b)]
    rate_by_index = {}
    for pos, d in zip(DIGIT_POSITIONS, digit_values):
        rate = params['r_base'] + d * params['r_step']
        for fid in groups[pos]:
            rate_by_index[flyid2i[str(fid)]] = rate

    for slot_idx, boost in recall_rate_by_slot.items():
        rate_by_index[slot_idx] = rate_by_index.get(slot_idx, 0 * Hz) + boost

    pois = _make_poisson_inputs_variable(neu, rate_by_index, params)
    net = Network(neu, syn, lateral_inh, spk_mon, *pois)
    net.run(params['chunk_dt'])

    counts = np.asarray(spk_mon.count[:])

    # --- MOTOR: decode ---
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

    # --- class-differentiated plasticity (sensory / motor / memory / other) ---
    neuron_class = np.full(len(neu), 'other', dtype=object)
    sensory_idx = [flyid2i[str(s)] for s in NEU_SUGAR if str(s) in flyid2i]
    for idx in sensory_idx:
        neuron_class[idx] = 'sensory'
    for idx in motor_indices:
        neuron_class[idx] = 'motor'
    for idx in active_memory.values():
        neuron_class[idx] = 'memory'
    for g in grown:
        if g['index'] < len(neuron_class) and g['parent'] < len(neuron_class):
            neuron_class[g['index']] = neuron_class[g['parent']]

    syn_i = np.asarray(syn.i[:])
    syn_j = np.asarray(syn.j[:])
    pre_class = neuron_class[syn_i]
    post_class = neuron_class[syn_j]
    class_mult = np.ones(len(syn_i))
    class_mult[post_class == 'motor'] = params['motor_mult']
    class_mult[post_class == 'memory'] = params['memory_mult']
    class_mult[(pre_class == 'sensory') & (post_class == 'sensory')] = params['sensory_mult']

    w_new, elig, wmax = update_weights_classed(syn, dopamine, class_mult, params)

    # --- mutegen (motor layer only, unaffected by memory) ---
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

    # --- MEMORY: reward for referenced neurons (separate local dopamine, own synapses only) ---
    memory_log = {}
    for key, idx in active_memory.items():
        bank = memory_bank[key]
        if referenced[key]:
            mem_dopamine = best_similarity[key] * params['memory_dopamine_gain']
            local_mask = (syn_j == idx).astype(float)
            update_weights_classed(syn, mem_dopamine, local_mask, params)
            bank['no_reference_streak'] = 0
            bank['last_referenced_trial'] = chunk_index
        else:
            bank['no_reference_streak'] += 1
        memory_log[key] = {'role': bank['role'], 'similarity': best_similarity[key],
                            'referenced': referenced[key], 'recalled_digit': recalled_digit[key],
                            'no_reference_streak': bank['no_reference_streak'],
                            'n_stored': len(bank['embeddings'])}

    # --- MEMORY: write (only neurons that actually fired this trial) ---
    full_vec = episode_embedding(a, b, true_digit, predicted_digit, correct)
    for key, idx in active_memory.items():
        if counts[idx] <= 0:
            continue
        bank = memory_bank[key]
        bank['embeddings'].append(full_vec.tolist())
        bank['meta'].append({'trial': chunk_index, 'true_digit': true_digit,
                              'predicted_digit': predicted_digit, 'correct': bool(correct)})
        cap = short_capacity_entries(params) if bank['role'] == 'short' else params['long_capacity_entries']
        while len(bank['embeddings']) > cap:
            bank['embeddings'].pop(0)
            bank['meta'].pop(0)
        memory_log[key]['n_stored'] = len(bank['embeddings'])
        memory_log[key]['wrote'] = True

    n_original_matched = sum(1 for fid in flyid2i if fid in annotations)
    n_memory_short = sum(1 for b in memory_bank.values() if b['role'] == 'short')
    n_memory_long = sum(1 for b in memory_bank.values() if b['role'] == 'long')
    n_memory_graduated = sum(1 for b in memory_bank.values() if b['role'] == 'graduated')
    n_memory_dead = sum(1 for b in memory_bank.values() if b['role'] == 'dead')
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
        'n_memory_short': n_memory_short, 'n_memory_long': n_memory_long,
        'n_memory_graduated': n_memory_graduated, 'n_memory_dead': n_memory_dead,
    }

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'connectome_id': conn_id,
        'connectome_state': connectome_state,
        'set_index': args.set_index,
        'trial_index': args.trial_index,
        'chunk_index': chunk_index,
        'operand_a': a, 'operand_b': b, 'true_last_digit': true_digit,
        'predicted_last_digit': predicted_digit, 'correct': correct,
        'dopamine': dopamine, 'mutated_digit_slot': mutated_digit,
        'memory': memory_log,
        'n_neurons': len(neu),
        'chunk_total_spikes': chunk_total_spikes,
        'n_active_synapses': n_active_synapses,
        'mean_abs_w_mV': float(np.mean(np.abs(w_new))),
        'mean_abs_elig_mV': float(np.mean(np.abs(elig))),
        'memory_events': [],
        'recall_events': recall_events,
        'recall_matched_true_digit': any(e['recalled_digit'] == true_digit for e in recall_events) if recall_events else None,
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

        # --- Short -> Long: real neurogenesis event, not decay ---
        for key in list(active_memory.keys()):
            bank = memory_bank[key]
            if bank['role'] != 'short':
                continue
            if bank['no_reference_streak'] < params['short_atrophy_patience']:
                continue
            short_idx = memory_index_map[key]
            state_snap = snapshot_state(neu, syn)
            neu, syn, spk_mon, new_indices = rebuild_model_with_growth(state_snap, params, [short_idx])
            new_idx = new_indices[0]
            grown.append({'index': int(new_idx), 'parent': int(short_idx), 'chunk': chunk_index})
            sat_streak = np.concatenate([sat_streak, np.zeros(1)])

            mem_long_counter += 1
            new_key = 'mem_long_{}'.format(mem_long_counter)
            memory_bank[new_key] = {
                'role': 'long', 'embeddings': list(bank['embeddings']), 'meta': list(bank['meta']),
                'created_trial': chunk_index, 'no_reference_streak': 0,
                'last_referenced_trial': chunk_index, 'transformed_from': key,
            }
            memory_index_map[new_key] = int(new_idx)
            bank['role'] = 'graduated'
            del memory_index_map[key]
            result['memory_events'].append({'type': 'short_to_long', 'from': key, 'to': new_key,
                                             'new_index': int(new_idx)})
            result['n_neurons'] = len(neu)

        # --- Long death: real pruning, only if genuinely never referenced ---
        prune_keys = [key for key, idx in memory_index_map.items()
                      if memory_bank.get(key, {}).get('role') == 'long'
                      and memory_bank[key]['no_reference_streak'] >= params['long_die_patience']]
        if prune_keys:
            prune_idx = [memory_index_map[k] for k in prune_keys]
            state_snap = snapshot_state(neu, syn)
            state_snap['sat_streak'] = sat_streak[:len(neu)]
            new_state, old_to_new = prune_neurons(state_snap, prune_idx)
            neu, syn, spk_mon = build_from_state(new_state, params)
            sat_streak = new_state['sat_streak']
            flyid2i, grown, memory_index_map = remap_indices(old_to_new, flyid2i, grown, memory_index_map)
            for key in prune_keys:
                memory_bank[key]['role'] = 'dead'
            result['memory_events'].append({'type': 'long_death', 'keys': prune_keys})
            result['n_neurons'] = len(neu)

    final_state = snapshot_state(neu, syn)
    final_state['sat_streak'] = sat_streak[:len(neu)]
    save_state(final_state, args.state)

    true_digit_hist[true_digit] += 1
    meta_out = {'flyid2i': flyid2i, 'n_original': n_original, 'grown': grown,
                'chunk_index': chunk_index + 1, 'true_digit_hist': true_digit_hist,
                'wrong_streak': wrong_streak, 'memory_index_map': memory_index_map,
                'mem_long_counter': mem_long_counter}
    with open(args.meta, 'w') as f:
        json.dump(meta_out, f)
    with open(args.memory_bank, 'w') as f:
        json.dump(memory_bank, f)

    os.makedirs(args.log_dir, exist_ok=True)
    trial_log_path = os.path.join(
        args.log_dir, 'memory_flywire{}_set{:02d}_trial{:03d}.json'.format(
            FLYWIRE_MATERIALIZATION, args.set_index, args.trial_index))
    with open(trial_log_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.log_dir, 'memory_run_log.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(json.dumps({k: result[k] for k in result if k != 'memory'}))
    print(json.dumps({'memory': memory_log}))


if __name__ == '__main__':
    main()
