# SPDX-License-Identifier: Apache-2.0
# Frozen previously validated MoT range conversion, without training imports.
import numpy as np
METADATA_NAMES = (
    'mot_document_ids',
    'mot_doc_mod_ids',
    'mot_image_gen_indicators',
    'mot_is_image_token',
    'mot_token_pos',
    'mot_dup_boundary',
)

def ranges_from_metadata(metadata, sequence_length):
    arrays = [np.asarray(metadata[name])[:sequence_length] for name in METADATA_NAMES]
    if any(a.ndim != 1 or len(a) != sequence_length for a in arrays):
        raise ValueError('MoT metadata must contain one value per sequence token.')
    doc, mod, gen, image, pos, dup = arrays
    boundaries = np.zeros(sequence_length + 1, dtype=bool)
    boundaries[0] = boundaries[-1] = True
    if sequence_length > 1:
        for values in (doc, mod, gen, image, dup):
            boundaries[1:-1] |= values[1:] != values[:-1]
        increments = pos[1:] - pos[:-1]
        boundaries[1:-1] |= (increments != 0) & (increments != 1)
        # Image tokens often share one temporal position. Keep those tokens
        # in a single constant-position run instead of one run per pixel.
        boundaries[2:-1] |= increments[1:] != increments[:-1]
    edges = np.flatnonzero(boundaries).tolist()
    runs = list(zip(edges[:-1], edges[1:]))
    slices = []

    def append(qs, qe, ks, ke, kind):
        if qs < qe and ks < ke:
            slices.append((int(qs), int(qe), int(ks), int(ke), kind))

    def lower(qs, qe, ks, ke, strict=False):
        # q_position >= k_position (+1 for strict inequality).
        qp = int(pos[qs]) - int(strict)
        kp = int(pos[ks])
        q_linear = qe - qs > 1 and pos[qs + 1] != pos[qs]
        k_linear = ke - ks > 1 and pos[ks + 1] != pos[ks]
        if not q_linear and not k_linear:
            if qp >= kp:
                append(qs, qe, ks, ke, 0)
            return
        if not q_linear:
            append(qs, qe, ks, ks + min(max(qp - kp + 1, 0), ke - ks), 0)
            return
        if not k_linear:
            append(qs + max(kp - qp, 0), qe, ks, ke, 0)
            return
        kend = kp + ke - ks
        lo, hi = max(qp, kp), min(qp + qe - qs, kend)
        append(qs + lo - qp, qs + hi - qp, ks, ks + hi - kp, 1)
        tail = max(qp, kend)
        append(qs + tail - qp, qe, ks, ke, 0)

    def upper_strict(qs, qe, ks, ke):
        # q_position < k_position, represented with top-left inverse causal.
        qp = int(pos[qs]) + 1
        kp = int(pos[ks])
        q_linear = qe - qs > 1 and pos[qs + 1] != pos[qs]
        k_linear = ke - ks > 1 and pos[ks + 1] != pos[ks]
        if not q_linear and not k_linear:
            if qp <= kp:
                append(qs, qe, ks, ke, 0)
            return
        if not q_linear:
            append(qs, qe, ks + max(qp - kp, 0), ke, 0)
            return
        if not k_linear:
            append(qs, qs + min(max(kp - qp + 1, 0), qe - qs), ks, ke, 0)
            return
        qend, kend = qp + qe - qs, kp + ke - ks
        append(qs, qs + min(qend, kp) - qp, ks, ke, 0)
        lo, hi = max(qp, kp), min(qend, kend)
        append(qs + lo - qp, qs + hi - qp, ks + lo - kp, ke, 2)

    for qs, qe in runs:
        for ks, ke in runs:
            same_mod = mod[qs] == mod[ks]
            if dup[ks] or (gen[ks] and not (gen[qs] and same_mod)):
                continue
            exclude_diagonal = bool(gen[qs] and not gen[ks])
            if image[qs] and same_mod:
                if exclude_diagonal:
                    lower(qs, qe, ks, ke, strict=True)
                    upper_strict(qs, qe, ks, ke)
                else:
                    append(qs, qe, ks, ke, 0)
            elif doc[qs] == doc[ks]:
                lower(qs, qe, ks, ke, strict=exclude_diagonal)

    # Coalesce adjacent K ranges with the same Q interval. A full prefix may
    # extend a bottom-right causal slice without changing its diagonal.
    merged = []
    for current in sorted(slices, key=lambda s: (s[0], s[1], s[2], s[3])):
        if merged:
            prev = merged[-1]
            if prev[:2] == current[:2] and prev[3] == current[2]:
                if prev[4] == 0 and current[4] in (0, 1):
                    merged[-1] = (*prev[:3], current[3], current[4])
                    continue
                if prev[4] == 2 and current[4] == 0:
                    merged[-1] = (*prev[:3], current[3], 2)
                    continue
        merged.append(current)
    result = np.asarray(merged, dtype=np.int32).reshape(-1, 5)
    return result[:, :2].copy(), result[:, 2:4].copy(), result[:, 4].copy()

