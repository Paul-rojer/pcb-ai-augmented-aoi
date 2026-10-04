"""
multi_golden.py
---------------
Multi-product, multi-reference golden-template inspector.

- One golden template PER board type (blue + green designs), discovered
  automatically by registration clustering of the good boards.
- Each type keeps SEVERAL good references (all warped into one common frame).
  A test board registers if it matches ANY of them -> fewer registration
  failures -> more good boards auto-cleared.
- Scoring: z = |aligned - golden_mean| / (golden_std + k); image score = mean of
  the top-0.05% z. Decision: PASS / REVIEW / FAIL (auto_fail optional).
- Routing: a board goes to the type it matches best; matches none -> REVIEW.

Also provides consensus_decision() to fuse the AOI machine verdict with the AI.
"""
import glob
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "roi"))
from aoi_profile_crop import crop_image, needs_crop, MIN_FRAC, MAX_FRAC  # noqa: E402

REF_SIZE = 1024
ORB_FEATURES = 3000
MIN_INLIERS = 30
STD_FLOOR = 12.0
BLUR_SIGMA = 2.0
BORDER = 50
TOP_FRAC = 0.0005
MAX_REFS = 1            # references per type. 1 = single seed reference (most
#                         accurate alignment). Multi-ref (>1) was tested and
#                         REDUCED accuracy: alternate refs carry residual
#                         misalignment that inflates anomaly scores. Keep at 1.


def consensus_decision(ai_decision, aoi_verdict):
    """Fuse AOI machine verdict with AI decision.
        AOI good + AI PASS       -> PASS
        AOI defect + AI not-PASS -> FAIL
        disagreement             -> REVIEW
    """
    ai_pass = (ai_decision == "PASS")
    aoi_pass = str(aoi_verdict).strip().lower() in ("good", "pass", "g", "ok")
    if aoi_pass and ai_pass:
        return "PASS"
    if (not aoi_pass) and (not ai_pass):
        return "FAIL"
    return "REVIEW"


def _orb():
    return cv2.ORB_create(ORB_FEATURES)


def _to_ref_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(cv2.resize(img, (REF_SIZE, REF_SIZE)), cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


class Template:
    def __init__(self, refs, name="", mean=None, std=None,
                 tau_pass=None, tau_fail=None):
        # refs: list of gray uint8 images, ALL in the same coordinate frame
        self.name = name
        self.refs = [r.astype(np.uint8) for r in refs]
        self.ref = self.refs[0]
        self.mean = mean
        self.std = std
        self.tau_pass = tau_pass
        self.tau_fail = tau_fail
        self.orb = _orb()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._feats = [self.orb.detectAndCompute(r, None) for r in self.refs]

    def _homography_to(self, idx, g):
        kp1, des1 = self._feats[idx]
        if des1 is None:
            return None, 0
        kp2, des2 = self.orb.detectAndCompute(g, None)
        if des2 is None:
            return None, 0
        m = sorted(self.bf.match(des1, des2), key=lambda x: x.distance)[:300]
        if len(m) < 20:
            return None, len(m)
        src = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None, 0
        return H, int(mask.sum())

    def match(self, g):
        """Best inlier count across all references (fast routing)."""
        best = 0
        for i in range(len(self.refs)):
            _, inl = self._homography_to(i, g)
            if inl > best:
                best = inl
        return best

    def register(self, g, ecc=True):
        g = _to_ref_gray(g)
        bestH, bi = None, 0
        for i in range(len(self.refs)):
            H, inl = self._homography_to(i, g)
            if inl > bi:
                bi, bestH = inl, H
        if bestH is None:
            return None, None, bi
        a = cv2.warpPerspective(g, bestH, (REF_SIZE, REF_SIZE)).astype(np.float32)
        valid = a > 0
        if valid.sum() < 1000:
            return None, None, bi
        if ecc and self.mean is not None:
            try:
                w = np.eye(2, 3, dtype=np.float32)
                _, w = cv2.findTransformECC(
                    self.mean.astype(np.uint8), a.astype(np.uint8), w,
                    cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-4), None, 5)
                a = cv2.warpAffine(a, w, (REF_SIZE, REF_SIZE))
                valid = a > 0
            except cv2.error:
                pass
        am, asd = a[valid].mean(), a[valid].std() + 1e-6
        a = (a - am) / asd * self.ref[valid].std() + self.ref[valid].mean()
        return a, valid, bi

    def score(self, aligned, valid):
        z = np.abs(aligned - self.mean) / (self.std + STD_FLOOR)
        # 3x3 Gaussian (matches the FPGA hardware kernel exactly) so the software
        # and FPGA engines produce identical scores. Replaces the wider cv2
        # GaussianBlur(sigma=2) used previously.
        z = cv2.filter2D(z, -1, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                          np.float32) / 16.0)
        m = np.zeros_like(valid)
        m[BORDER:REF_SIZE - BORDER, BORDER:REF_SIZE - BORDER] = True
        zz = z[valid & m]
        s = float(np.mean(np.sort(zz)[-int(TOP_FRAC * zz.size):]))
        return s, z * (valid & m)

    def decide(self, s):
        if self.tau_pass is not None and s <= self.tau_pass:
            return "PASS"
        if self.tau_fail is not None and s >= self.tau_fail:
            return "FAIL"
        return "REVIEW"


class MultiGolden:
    def __init__(self, gallery_dir=None, auto_fail=False):
        self.templates = []
        self.auto_fail = auto_fail
        if gallery_dir and os.path.isdir(gallery_dir):
            self.load(gallery_dir)

    def inspect(self, bgr, do_crop="auto", engine="software"):
        if do_crop == "auto":
            do_crop = needs_crop(bgr)
        if do_crop:
            bgr, frac = crop_image(bgr)
            if frac < MIN_FRAC or frac > MAX_FRAC:
                return {"decision": "REVIEW", "reason": "ROI crop low-confidence",
                        "score": None, "type": None}, None
        g = _to_ref_gray(bgr)
        best = None
        for t in self.templates:
            inl = t.match(g)
            if inl >= MIN_INLIERS and (best is None or inl > best[0]):
                best = (inl, t)
        if best is None:
            return {"decision": "REVIEW", "reason": "no matching board type (unknown product)",
                    "score": None, "type": None}, None
        inl, t = best
        a, valid, _ = t.register(g, ecc=True)
        if a is None:
            return {"decision": "REVIEW", "reason": "registration failed",
                    "score": None, "type": t.name}, None
        hw_metrics = None
        if engine == "fpga":
            import os as _os, sys as _sys
            _f = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "fpga")
            if _f not in _sys.path:
                _sys.path.insert(0, _f)
            import hw_emulator as _hw
            m8, rcp = _hw.prepare_golden(t.mean, t.std)
            s, z, hw_metrics = _hw.run(a, m8, rcp, valid, border=BORDER)
        else:
            s, z = t.score(a, valid)
        decision = t.decide(s)
        if not self.auto_fail and decision == "FAIL":
            decision = "REVIEW"
        out = {"decision": decision, "score": round(s, 3), "type": t.name,
               "inliers": inl, "reason": "", "engine": engine}
        if hw_metrics:
            out["hw_metrics"] = hw_metrics
        return out, z

    # ---------- build ----------
    def build(self, image_paths, min_type_size=3, match_inliers=40,
              max_refs=MAX_REFS, verbose=True):
        orb = _orb()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def feats(g):
            return orb.detectAndCompute(g, None)

        def inliers(fa, fb):
            (kp1, des1), (kp2, des2) = fa, fb
            if des1 is None or des2 is None:
                return None, 0
            m = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:300]
            if len(m) < 20:
                return None, 0
            src = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
            dst = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            return (H, int(mask.sum())) if H is not None else (None, 0)

        # load all good boards once
        boards = []
        for p in image_paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            g = _to_ref_gray(im)
            boards.append((g, feats(g)))

        # greedy clustering by registration
        clusters = []  # each: {seed_idx, members:[idx]}
        for i, (g, f) in enumerate(boards):
            best, bi = None, 0
            for ci, c in enumerate(clusters):
                _, inl = inliers(boards[c["seed"]][1], f)
                if inl > bi:
                    bi, best = inl, ci
            if bi >= match_inliers:
                clusters[best]["members"].append(i)
            else:
                clusters.append({"seed": i, "members": [i]})
        clusters.sort(key=lambda c: -len(c["members"]))

        self.templates = []
        for ci, c in enumerate(clusters):
            mem = c["members"]
            if len(mem) < min_type_size:
                if verbose:
                    print(f"  skip cluster ({len(mem)} boards < {min_type_size})")
                continue
            seed_g, seed_f = boards[c["seed"]]
            # align every member to the seed frame; build mean/std
            stack = []
            for j in mem:
                gj, fj = boards[j]
                H, inl = inliers(seed_f, fj)
                if H is None or inl < MIN_INLIERS:
                    continue
                a = cv2.warpPerspective(gj, H, (REF_SIZE, REF_SIZE)).astype(np.float32)
                if (a > 0).sum() > 1000:
                    stack.append(a)
            if len(stack) < min_type_size:
                continue
            stack = np.array(stack, dtype=np.float32)
            mean, std = stack.mean(0), stack.std(0)
            # references: seed + up to (max_refs-1) evenly-spaced members, in seed frame
            ref_imgs = [seed_g]
            others = [j for j in mem if j != c["seed"]]
            step = max(1, len(others) // max(1, max_refs - 1))
            for j in others[::step][:max_refs - 1]:
                gj, fj = boards[j]
                H, inl = inliers(seed_f, fj)
                if H is None or inl < MIN_INLIERS:
                    continue
                aligned = cv2.warpPerspective(gj, H, (REF_SIZE, REF_SIZE))
                ref_imgs.append(aligned.astype(np.uint8))
            t = Template(ref_imgs, name=f"type{ci}", mean=mean, std=std)
            sc = [t.score(a, a > 0)[0] for a in stack]
            t.tau_pass = float(np.percentile(sc, 90) + 0.45)
            t.tau_fail = float(t.tau_pass * 1.25)
            self.templates.append(t)
            if verbose:
                print(f"  {t.name}: {len(stack)} boards, {len(ref_imgs)} refs, "
                      f"tau_pass={t.tau_pass:.2f}")
        return self

    # ---------- persistence ----------
    def save(self, gallery_dir):
        os.makedirs(gallery_dir, exist_ok=True)
        for t in self.templates:
            np.savez(os.path.join(gallery_dir, f"{t.name}.npz"),
                     refs=np.stack(t.refs), mean=t.mean, std=t.std,
                     tau_pass=t.tau_pass, tau_fail=t.tau_fail)

    def load(self, gallery_dir):
        self.templates = []
        for f in sorted(glob.glob(os.path.join(gallery_dir, "*.npz"))):
            try:
                d = np.load(f)
                name = os.path.splitext(os.path.basename(f))[0]
                refs = list(d["refs"]) if "refs" in d.files else [d["ref"]]
                self.templates.append(Template(
                    refs, name=name, mean=d["mean"], std=d["std"],
                    tau_pass=float(d["tau_pass"]), tau_fail=float(d["tau_fail"])))
            except Exception as e:
                print(f"[warn] skipping unreadable template {f}: {e}")
        return self
