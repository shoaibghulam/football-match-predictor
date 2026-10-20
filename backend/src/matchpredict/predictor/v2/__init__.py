"""MatchPredict model v2: one global rating for domestic + European club football.

Modules:

* ``names``          canonical team names / countries across all sources
* ``data``           unified, deduplicated match frame (domestic + UCL/UEL/UECL) from the DB
* ``global_rating``  own cross-country Elo + goal ratings (country offsets linked by UEFA results)
* ``features``       one feature table for every match (global + v1 domestic features + flags)
* ``ensemble``       log-linear pooling / temperature calibration
* ``train``          walk-forward evaluation, ablations, promotion gate, final artifact
* ``service``        ``V2Service`` (contracts.PredictionService, Europe-capable)
* ``retrain``        weekly retrain with a no-regression promotion check
"""
