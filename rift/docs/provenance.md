# Implementation credits

RIFT extends the ECEformer/TKGElib codebase and inherited LibKGE/HittER
infrastructure. The original source notices are retained in the implementation
and `ThirdPartyNotices.txt`.

- [LibKGE](https://github.com/uma-pi1/kge): configuration, dataset, training,
  evaluation, and embedding-framework infrastructure.
- [Microsoft HittER](https://github.com/microsoft/HittER): Transformer-based KGE
  infrastructure and inherited MIT notice.
- Hugging Face: the legacy BERT encoder dependency and its accompanying notice.
- Additional source references in `kge/util/rat.py` and `kge/util/sc.py` are
  retained with those utilities.

RIFT-specific model code is in `kge/model/rift.py`; temporal retrieval code is in
`kge/model/causal_context.py`. Keep these acknowledgements when redistributing
the implementation.
