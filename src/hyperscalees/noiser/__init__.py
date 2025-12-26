from . import base_noiser, open_es, eggroll, alteggroll, sparse, eggroll_baseline_subtraction

all_noisers = {
    "noop": base_noiser.Noiser,
    "open_es": open_es.OpenES,
    "eggroll": eggroll.EggRoll,
    "eggrollbs": eggroll_baseline_subtraction.EggRollBS,
    "alteggroll": alteggroll.EggRoll,
    "reeggroll": eggroll.EggRoll,
    "sparse": sparse.Sparse
}
