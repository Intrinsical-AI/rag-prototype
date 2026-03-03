import time
import random
import string

def generate_eids(n):
    return [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(n)]

def bench_original(queries, k):
    hits = 0
    rr_sum = 0.0
    for q_relevant, retrieved_ext in queries:
        relevant = set(q_relevant)
        hit = any(eid in relevant for eid in retrieved_ext)
        if hit:
            hits += 1

        rank = None
        for i, eid in enumerate(retrieved_ext, 1):
            if eid in relevant:
                rank = i
                break
        if rank is not None:
            rr_sum += 1.0 / float(rank)
    return hits, rr_sum

def bench_optimized(queries, k):
    hits = 0
    rr_sum = 0.0
    for q_relevant, retrieved_ext in queries:
        relevant = set(q_relevant)
        rank = None
        for i, eid in enumerate(retrieved_ext, 1):
            if eid in relevant:
                rank = i
                break
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / float(rank)
    return hits, rr_sum

queries = []
for _ in range(10000):
    rel = generate_eids(5)
    ret = generate_eids(50)
    # Add some overlap to make hits happen
    if random.random() > 0.5:
        ret[random.randint(0, 49)] = rel[random.randint(0, 4)]
    queries.append((rel, ret))

start = time.time()
bench_original(queries, 50)
end = time.time()
print(f"Original: {end - start:.4f}s")

start = time.time()
bench_optimized(queries, 50)
end = time.time()
print(f"Optimized: {end - start:.4f}s")
