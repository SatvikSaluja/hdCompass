from hdcompass.align import align, angle_error
from hdcompass.features import population_rates
from hdcompass.manifold import embed, ring_angle


def test_unsupervised_ring_recovers_angle(hd40):
    spikes, angle, _ = hd40
    rates = population_rates(spikes, spikes.time_support)
    emb = embed(rates)
    rec = ring_angle(emb)
    assert len(rec) == len(rates) and (rec.t == rates.t).all()
    aligned, _ = align(rec, angle)
    assert angle_error(aligned, angle) < 0.35
