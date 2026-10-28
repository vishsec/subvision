import statistics

import serve

raw = open("data/test/3/9011053L.png", "rb").read()

serve.predict_raw(raw)  # warmup (first run includes lazy init)

totals = []
infers = []
for _ in range(50):
    r = serve.predict_raw(raw)
    totals.append(r["latency_ms"]["total"])
    infers.append(r["latency_ms"]["inference"])


def p(xs, q):
    xs = sorted(xs)
    return xs[int(len(xs) * q) - 1]


print(f"n={len(totals)}  (CPU, single image, in-process)")
print(f"total   p50={p(totals,0.5):.1f}ms  p95={p(totals,0.95):.1f}ms  mean={statistics.mean(totals):.1f}ms")
print(f"infer   p50={p(infers,0.5):.1f}ms  p95={p(infers,0.95):.1f}ms  mean={statistics.mean(infers):.1f}ms")
