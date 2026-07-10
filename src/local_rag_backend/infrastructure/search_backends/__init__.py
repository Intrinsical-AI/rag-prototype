"""Search backend adapters for structured retrieval."""

from .elastic_like import ElasticLikeSearchRetriever
from .local_split import LocalSplitSearchRetriever
from .solr import SolrSearchRetriever

__all__ = [
    "ElasticLikeSearchRetriever",
    "LocalSplitSearchRetriever",
    "SolrSearchRetriever",
]
