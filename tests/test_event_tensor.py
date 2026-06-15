from causal_smart_home.demo_data import make_toy_normal_sequences
from causal_smart_home.event_tensor import EventTensorizer


def test_event_tensor_shape_and_channels():
    seqs = make_toy_normal_sequences(5)
    tensorized = EventTensorizer(level="action", count_mode="binary").fit_transform(seqs)
    assert tensorized.tensor.shape[0] == 5 * 56
    assert tensorized.tensor.shape[1] >= 7
    assert "a:10" in tensorized.key_to_channel
