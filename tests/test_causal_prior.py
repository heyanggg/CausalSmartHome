from causal_smart_home.demo_data import make_toy_normal_sequences
from causal_smart_home.event_tensor import EventTensorizer
from causal_smart_home.causal_prior import GradientCausalMiner


def test_gradient_causal_prior_serializable():
    seqs = make_toy_normal_sequences(20)
    tx = EventTensorizer(level="action", count_mode="binary", decay=0.1).fit_transform(seqs)
    prior = GradientCausalMiner(lag=3, epochs=2, hidden=16, batch_size=16).fit_prior(tx.tensor, tx.channel_to_key, sample_limit=20)
    assert len(prior.matrix) == len(tx.channel_to_key)
    assert prior.top_edges(k=3) is not None
