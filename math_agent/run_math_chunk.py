'''Advance the partial-connectome plasticity model by exactly one chunk,
gated by an externally-supplied reward (correct/incorrect + response time on
a math problem), then persist the resulting state to disk and write a full
provenance log of the trial.

This is the bridge between the Brian2 simulation (a single Python process
per invocation) and an outer loop driven by an orchestrator that spawns a
separate AI agent each trial: the orchestrator calls this script once per
trial with --correct 1/0 and --response-ms, and the connectome's state
(voltages, weights, eligibility traces, growth ceilings, saturation streaks,
grown-neuron log) carries over via --state/--meta between calls.
'''

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brian2 import ms, mV, Network

from plasticity import (
    create_plastic_model, save_state, load_state, build_from_state,
    snapshot_state, apply_growth_atrophy, find_growth_candidates,
    rebuild_model_with_growth, update_weights, external_reward_dopamine_timed,
    plastic_params, _make_poisson_inputs,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_COMP = os.path.join(_HERE, 'subgraph_comp_1hop_783.csv')
DEFAULT_CON = os.path.join(_HERE, 'subgraph_con_1hop_783.parquet')
DEFAULT_LOG_DIR = r'C:\Users\caele\OneDrive\Desktop\Project\Drosophila_brain_model\Test Logs'
ANNOTATIONS_PATH = os.path.join(_REPO_ROOT, 'annotations', 'flywire_783_neuron_annotations.tsv')
FLYWIRE_MATERIALIZATION = '783'

# 19 of the original 21 labellar gustatory (LB3) sensory neurons from the paper's tutorial that
# still carry the same root_id AND actually exist in materialization 783's Completeness_783.csv
# (2 were superseded/absent by proofreading since 630 -- one of them, 720575940620900446, is
# present in the annotation table but not in Completeness_783.csv itself)
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
]


def load_annotations():
    '''flyid(str) -> {super_class, cell_class, cell_type, side} from the FlyWire
    783 community annotation table (flyconnectome/flywire_annotations).'''
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


def build_params():
    params = dict(plastic_params)
    params['chunk_dt'] = 20 * ms
    params['growth_mult'] = 1.3
    params['w_max_floor'] = 0.2 * mV
    params['sat_frac_thr'] = 0.5
    params['sat_patience'] = 2
    params['lr'] = 0.2
    params['max_new_neurons'] = 10
    params['penalty'] = -1.0        # punish mistakes as hard as success rewards
    params['deadline_ms'] = 15000   # real time-pressure deadline for the AI agent's response
    params['speed_gain'] = 0.6
    return params


def file_hash(path, n=8):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:n]


def connectome_id(comp_path, con_path):
    return 'flywire{}_1hop_sugar_mn9_comp-{}_con-{}'.format(
        FLYWIRE_MATERIALIZATION, file_hash(comp_path), file_hash(con_path))


def label_for(idx, i2flyid, grown_by_index):
    '''Flywire ID for an original connectome neuron, or a synthetic
    grown-lineage label (grown:<index>:parent=<label of its parent>) for a
    neurogenesis-spawned one.'''
    if idx in i2flyid:
        return i2flyid[idx]
    parent = grown_by_index.get(idx)
    if parent is None:
        return 'unknown:{}'.format(idx)
    return 'grown:{}:parent={}'.format(idx, label_for(parent, i2flyid, grown_by_index))


def annotate(flyid, annotations):
    '''Cell-type annotation dict for a Flywire ID, or a marker dict if it's a
    grown (synthetic) neuron or has no entry in the annotation table.'''
    if flyid.startswith('grown:') or flyid.startswith('unknown:'):
        return {'super_class': 'grown', 'cell_class': None, 'cell_type': None, 'side': None}
    return annotations.get(flyid, {'super_class': 'unmatched', 'cell_class': None,
                                    'cell_type': None, 'side': None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True, help='npz path for persisted network state')
    ap.add_argument('--meta', required=True, help='json path for persisted run metadata')
    ap.add_argument('--comp', default=DEFAULT_COMP)
    ap.add_argument('--con', default=DEFAULT_CON)
    ap.add_argument('--correct', type=int, required=True, choices=[0, 1])
    ap.add_argument('--response-ms', type=float, required=True)
    ap.add_argument('--review-every', type=int, default=3)
    ap.add_argument('--log-dir', default=DEFAULT_LOG_DIR)
    ap.add_argument('--set-index', type=int, default=0)
    ap.add_argument('--trial-index', type=int, default=0)
    ap.add_argument('--problem', default='', help='e.g. "4821 x 6790"')
    ap.add_argument('--ground-truth', default='')
    ap.add_argument('--agent-answer', default='')
    ap.add_argument('--timed-out', type=int, default=0, choices=[0, 1])
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
    else:
        neu, syn, spk_mon, df_comp = create_plastic_model(args.comp, args.con, params)
        n_original = len(df_comp)
        grown = []
        chunk_index = 0
        sat_streak = np.zeros(n_original + params['max_new_neurons'])
        flyid2i = {str(int(j)): int(i) for i, j in enumerate(df_comp.index)}

    i2flyid = {i: fid for fid, i in flyid2i.items()}
    grown_by_index = {g['index']: g['parent'] for g in grown}

    exc = [flyid2i[str(n)] for n in NEU_SUGAR]

    pois = _make_poisson_inputs(neu, exc, params['r_poi'], params)
    net = Network(neu, syn, spk_mon, *pois)
    net.run(params['chunk_dt'])

    # per-neuron spike counts THIS chunk (spk_mon is fresh each process invocation)
    counts = np.asarray(spk_mon.count[:])
    spiking_neurons = []
    for idx, c in enumerate(counts):
        if c <= 0:
            continue
        flyid = label_for(int(idx), i2flyid, grown_by_index)
        entry = {'id': flyid, 'spikes': int(c)}
        entry.update(annotate(flyid, annotations))
        spiking_neurons.append(entry)
    spiking_neurons.sort(key=lambda d: -d['spikes'])

    chunk_total_spikes = int(counts.sum())
    w_mV = np.asarray(syn.w[:] / mV)
    elig_mV = np.asarray(syn.elig[:] / mV)
    n_active_synapses = int(np.count_nonzero(np.abs(w_mV) >= (params['active_syn_thr'] / mV)))
    n_grown_neurons = len(grown)

    # synapses most implicated in this trial: highest |eligibility| this chunk
    top_k = min(15, len(elig_mV))
    top_idx = np.argsort(-np.abs(elig_mV))[:top_k]
    syn_i = np.asarray(syn.i[:])
    syn_j = np.asarray(syn.j[:])
    top_synapses = []
    for k in top_idx:
        if abs(elig_mV[k]) <= 0:
            continue
        pre_id = label_for(int(syn_i[k]), i2flyid, grown_by_index)
        post_id = label_for(int(syn_j[k]), i2flyid, grown_by_index)
        top_synapses.append({
            'pre': pre_id,
            'pre_cell_type': annotate(pre_id, annotations)['cell_type'],
            'post': post_id,
            'post_cell_type': annotate(post_id, annotations)['cell_type'],
            'w_mV': float(w_mV[k]),
            'elig_mV': float(elig_mV[k]),
        })

    correct = bool(args.correct) and not bool(args.timed_out)
    dopamine = external_reward_dopamine_timed(
        correct, args.response_ms, chunk_total_spikes,
        n_active_synapses, n_grown_neurons, params)
    w_new, elig, wmax = update_weights(syn, dopamine, params)

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

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'connectome_id': conn_id,
        'connectome_state': connectome_state,
        'set_index': args.set_index,
        'trial_index': args.trial_index,
        'chunk_index': chunk_index,
        'problem': args.problem,
        'ground_truth': args.ground_truth,
        'agent_answer': args.agent_answer,
        'raw_correct': bool(args.correct),
        'timed_out': bool(args.timed_out),
        'correct': correct,
        'response_ms': args.response_ms,
        'deadline_ms': params['deadline_ms'],
        'n_neurons': len(neu),
        'dopamine': dopamine,
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

    meta_out = {'flyid2i': flyid2i, 'n_original': n_original, 'grown': grown,
                'chunk_index': chunk_index + 1}
    with open(args.meta, 'w') as f:
        json.dump(meta_out, f)

    os.makedirs(args.log_dir, exist_ok=True)
    trial_log_path = os.path.join(
        args.log_dir, 'flywire{}_set{:02d}_trial{:03d}.json'.format(
            FLYWIRE_MATERIALIZATION, args.set_index, args.trial_index))
    with open(trial_log_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.log_dir, 'run_log.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(json.dumps({k: result[k] for k in result if k not in
                       ('spiking_neurons', 'top_synapses_by_eligibility')}))


if __name__ == '__main__':
    main()
