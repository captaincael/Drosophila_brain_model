'''Extract a small neighborhood subgraph from the full connectome so
reward/growth experiments can iterate in seconds instead of minutes.

BFS outward (in both directions) from a set of seed Flywire IDs through the
connectivity table, `n_hops` steps, and write out a completeness/connectivity
file pair in the same format `model.create_model` / `plasticity.create_plastic_model`
already expect.
'''

import pandas as pd


def extract_subgraph(path_comp, path_con, seed_ids, n_hops=1,
                      path_out_comp=None, path_out_con=None):
    '''Build a connectome subset reachable within `n_hops` hops of `seed_ids`.

    Parameters
    ----------
    path_comp, path_con : str
        paths to the full completeness csv / connectivity parquet
    seed_ids : iterable of int
        flywire IDs to grow the neighborhood from
    n_hops : int
        number of BFS hops (both pre- and post-synaptic direction)
    path_out_comp, path_out_con : str, optional
        if given, write the subset out in the same csv/parquet format as the
        full connectome files, so it can be loaded with the existing
        `create_model` / `create_plastic_model` unchanged

    Returns
    -------
    df_comp_sub, df_con_sub : pandas.DataFrame
    '''

    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    pre = df_con['Presynaptic_ID'].values
    post = df_con['Postsynaptic_ID'].values

    seeds = set(seed_ids)
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(n_hops):
        pre_hit = pd.Series(pre).isin(frontier).values
        post_hit = pd.Series(post).isin(frontier).values
        # neighbor of a frontier presynaptic neuron is its postsynaptic target, and vice versa
        neigh = set(post[pre_hit]) | set(pre[post_hit])
        new = neigh - visited
        if not new:
            break
        visited |= new
        frontier = new

    df_comp_sub = df_comp.loc[df_comp.index.isin(visited)].copy()

    edge_mask = pd.Series(pre).isin(visited).values & pd.Series(post).isin(visited).values
    df_con_sub = df_con.loc[edge_mask].copy()

    # re-index Presynaptic_Index/Postsynaptic_Index against the subset ordering
    flyid2i = {fid: i for i, fid in enumerate(df_comp_sub.index)}
    df_con_sub['Presynaptic_Index'] = df_con_sub['Presynaptic_ID'].map(flyid2i)
    df_con_sub['Postsynaptic_Index'] = df_con_sub['Postsynaptic_ID'].map(flyid2i)

    if path_out_comp:
        df_comp_sub.to_csv(path_out_comp)
    if path_out_con:
        df_con_sub.to_parquet(path_out_con, compression='brotli')

    return df_comp_sub, df_con_sub
