"""
fuzzy_weight.py — Upgraded Mamdani FIS for drug-disease prediction score refinement.

Removes database dependency. Works entirely with numpy arrays.
Implements a full Mamdani FIS with:
  - 3 antecedents: cf_score, src_neighbor, tgt_neighbor
  - Triangular membership functions: Low / Mid / High
  - 11 IF-THEN rules
  - 1 consequent: fuzzy_score
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class MamdaniFIS:
    """
    Mamdani Fuzzy Inference System for refining GCN drug-disease predictions.

    Inputs (all normalised to [0, 1]):
      cf_score     — GCN probability for the drug-disease pair
      src_neighbor — mean similarity of the drug to its top-K neighbours
      tgt_neighbor — mean similarity of the disease to its top-K neighbours

    Output:
      fuzzy_score  — refined association confidence in [0, 1]
    """

    def __init__(self):
        universe = np.linspace(0, 1, 101)

        # --- Antecedents ---
        cf  = ctrl.Antecedent(universe, 'cf_score')
        src = ctrl.Antecedent(universe, 'src_neighbor')
        tgt = ctrl.Antecedent(universe, 'tgt_neighbor')
        # --- Consequent ---
        out = ctrl.Consequent(universe, 'fuzzy_score')

        # Triangular MFs  [left, peak, right]
        for var in [cf, src, tgt, out]:
            var['low']  = fuzz.trimf(var.universe, [0.0, 0.0, 0.5])
            var['mid']  = fuzz.trimf(var.universe, [0.0, 0.5, 1.0])
            var['high'] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

        # --- 11 IF-THEN rules ---
        # Design principle:
        #   - Trust GCN fully when cf_score is HIGH (rare/novel pairs may have low neighbors)
        #   - Use neighbor evidence to resolve uncertainty when cf_score is MID
        #   - Keep output LOW only when both GCN and neighborhood evidence agree
        rules = [
            ctrl.Rule(cf['high'] & src['high'] & tgt['high'], out['high']),   # R1
            ctrl.Rule(cf['high'] & src['high'] & tgt['mid'],  out['high']),   # R2
            ctrl.Rule(cf['high'] & src['mid']  & tgt['high'], out['high']),   # R3
            ctrl.Rule(cf['high'] & src['mid']  & tgt['mid'],  out['high']),   # R4
            ctrl.Rule(cf['high'] & src['low']  & tgt['low'],  out['high']),   # R5 FIX: trust GCN confidence; novel drugs/diseases have low neighbors by nature
            ctrl.Rule(cf['mid']  & src['high'] & tgt['high'], out['high']),   # R6
            ctrl.Rule(cf['mid']  & src['mid']  & tgt['mid'],  out['mid']),    # R7
            ctrl.Rule(cf['mid']  & src['low']  & tgt['low'],  out['mid']),    # R8 FIX: uncertain GCN + no neighbor evidence → mid, not low
            ctrl.Rule(cf['low']  & src['high'] & tgt['high'], out['mid']),    # R9
            ctrl.Rule(cf['low']  & src['mid']  & tgt['mid'],  out['low']),    # R10
            ctrl.Rule(cf['low']  & src['low'],                out['low']),    # R11
        ]

        ctrl_sys = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(ctrl_sys)

        # Keep references for membership queries
        self._cf  = cf
        self._src = src
        self._tgt = tgt
        self._out = out

    # ------------------------------------------------------------------
    def compute(self, cf_score: float, src_neighbor: float, tgt_neighbor: float) -> float:
        """Return fuzzy refined score for a single pair."""
        self._sim.input['cf_score']     = float(np.clip(cf_score,     0.001, 0.999))
        self._sim.input['src_neighbor'] = float(np.clip(src_neighbor, 0.001, 0.999))
        self._sim.input['tgt_neighbor'] = float(np.clip(tgt_neighbor, 0.001, 0.999))
        self._sim.compute()
        return float(self._sim.output['fuzzy_score'])

    def compute_batch(self, cf_scores, src_neighbors, tgt_neighbors) -> np.ndarray:
        """Vectorised batch version (falls back to cf_score on error)."""
        results = []
        for cf, src, tgt in zip(cf_scores, src_neighbors, tgt_neighbors):
            try:
                results.append(self.compute(cf, src, tgt))
            except Exception:
                results.append(float(cf))
        return np.array(results, dtype=np.float32)

    def get_memberships(self, cf_score: float, src_neighbor: float, tgt_neighbor: float) -> dict:
        """
        Return a dict with raw input values, per-variable membership degrees,
        and the final fuzzy score.  Used by the frontend Fuzzy-Logic modal.
        """
        def _mem(var, value):
            v = float(np.clip(value, 0, 1))
            return {
                'lo':  round(float(fuzz.interp_membership(var.universe, var['low'].mf,  v)), 4),
                'mid': round(float(fuzz.interp_membership(var.universe, var['mid'].mf,  v)), 4),
                'hi':  round(float(fuzz.interp_membership(var.universe, var['high'].mf, v)), 4),
            }

        fuzzy_score = self.compute(cf_score, src_neighbor, tgt_neighbor)
        return {
            'cf_score':        round(cf_score,     4),
            'src_neighbor':    round(src_neighbor, 4),
            'tgt_neighbor':    round(tgt_neighbor, 4),
            'cf_memberships':  _mem(self._cf,  cf_score),
            'src_memberships': _mem(self._src, src_neighbor),
            'tgt_memberships': _mem(self._tgt, tgt_neighbor),
            'fuzzy_score':     round(fuzzy_score, 4),
            'num_rules':       11,
            'mf_type':         'triangular',
            'num_inputs':      3,
        }


# ── legacy entry-point ────────────────────────────────────────────────
def apply_fuzzy_logic():
    fis = MamdaniFIS()
    demo = fis.get_memberships(0.75, 0.60, 0.55)
    print(">>> ĐÃ SẴN SÀNG HỆ THỐNG FUZZY ĐỂ LỌC NHIỄU!")
    print(f"Demo output (cf=0.75, src=0.60, tgt=0.55): fuzzy_score={demo['fuzzy_score']}")
    return fis


if __name__ == "__main__":
    apply_fuzzy_logic()
