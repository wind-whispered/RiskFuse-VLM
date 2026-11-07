"""Caption-level quality: CHAIR-i, BLEU-4, ROUGE-L, object recall (CHAIR-500)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

import numpy as np

# COCO synonym table used by the original CHAIR implementation (Rohrbach et al.).
COCO_SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "girl", "boy", "man", "woman", "kid", "child", "chef", "baker",
               "people", "adult", "rider", "children", "baby", "worker", "passenger",
               "guy", "lady", "player", "men", "women", "spectator", "male", "female"],
    "bicycle": ["bicycle", "bike", "bicycles", "bikes", "cycle", "cycles"],
    "car": ["car", "cars", "suv", "minivan", "taxi", "jeep", "cab"],
    "motorcycle": ["motorcycle", "scooter", "motor bike", "motor cycle", "motorbike", "moped"],
    "airplane": ["airplane", "jetliner", "plane", "air plane", "aeroplane", "aircraft", "jet"],
    "bus": ["bus", "buses", "minibus", "trolley"],
    "train": ["train", "locomotive", "tramway", "caboose", "subway"],
    "truck": ["truck", "trucks", "pickup", "lorry"],
    "boat": ["boat", "ship", "canoe", "sailboat", "yacht", "kayak", "raft", "dinghy"],
    "traffic light": ["traffic light", "stop light", "traffic signal", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds", "ostrich", "owl", "seagull", "goose", "duck", "parakeet",
             "falcon", "robin", "pelican", "swan", "gull", "flamingo", "parrot", "pigeon"],
    "cat": ["cat", "cats", "kitten", "kitty"],
    "dog": ["dog", "dogs", "puppy", "beagle", "pup", "chihuahua", "schnauzer", "poodle"],
    "horse": ["horse", "horses", "pony", "colt", "foal", "stallion", "mare"],
    "sheep": ["sheep", "lamb", "ram", "goat", "ewe"],
    "cow": ["cow", "cows", "cattle", "oxen", "ox", "calf", "bull", "holstein"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears", "panda"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "knapsack", "rucksack"],
    "umbrella": ["umbrella", "umbrellas"],
    "handbag": ["handbag", "wallet", "purse", "briefcase"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "suit case", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball", "basketball", "football", "soccer ball",
                    "baseball", "volleyball", "tennis ball"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "longboard", "skimboard", "shortboard", "wakeboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wineglass"],
    "cup": ["cup", "cups", "mug"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives", "pocketknife", "knive"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls", "container"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches", "burger", "hamburger", "bun"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "doughnut", "donuts", "bagel"],
    "cake": ["cake", "cakes", "cheesecake", "cupcake"],
    "chair": ["chair", "chairs", "seat", "stool"],
    "couch": ["couch", "sofa", "loveseat", "recliner"],
    "potted plant": ["potted plant", "houseplant", "plant"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet", "urinal", "commode", "lavatory"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "computer", "notebook", "netbook", "macbook"],
    "mouse": ["mouse"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "mobile phone", "phone", "cellphone", "smartphone", "iphone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove", "stovetop"],
    "toaster": ["toaster"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors", "shears"],
    "teddy bear": ["teddy bear", "teddybear", "teddy"],
    "hair drier": ["hair drier", "hairdryer", "hair dryer"],
    "toothbrush": ["toothbrush"],
}
_SYN2CANON: Dict[str, str] = {
    syn: canon for canon, syns in COCO_SYNONYMS.items() for syn in syns
}
_SPECIAL_PHRASES: Dict[str, str] = {
    "street light": "traffic light",
    "bow tie": "tie",
    "toilet seat": "toilet",
    "laptop computer": "laptop",
    "stove top oven": "oven",
    "train track": "train",
    "wine glas": "wine glass",
    "passenger jet": "airplane",
    "passenger train": "train",
}
for _animal in (
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
):
    _SPECIAL_PHRASES[f"baby {_animal}"] = _animal
    _SPECIAL_PHRASES[f"adult {_animal}"] = _animal
_PHRASE2CANON = {**_SYN2CANON, **_SPECIAL_PHRASES}
_MAX_NGRAM = max(len(s.split()) for s in _PHRASE2CANON)


def mentioned_object_instances(caption: str) -> List[str]:
    """Canonical COCO object mentions, preserving repeated instances.

    The original CHAIR implementation counts ``node_words`` rather than a set.
    Greedy longest-match tokenisation also mirrors its treatment of COCO double
    words and prevents phrases such as ``laptop computer`` from being counted
    twice.
    """
    words = re.findall(r"[a-z]+", str(caption).lower())
    found: List[str] = []
    i = 0
    while i < len(words):
        matched = False
        for n in range(min(_MAX_NGRAM, len(words) - i), 0, -1):
            phrase = " ".join(words[i : i + n])
            canonical = _PHRASE2CANON.get(phrase)
            if canonical is None:
                continue
            found.append(canonical)
            i += n
            matched = True
            break
        if not matched:
            i += 1
    return found


def mentioned_objects(caption: str) -> Set[str]:
    """Canonical COCO objects mentioned in a caption (CHAIR-style n-gram match)."""
    return set(mentioned_object_instances(caption))


def mask_object_mentions(caption: str, canonical_objects: Sequence[str]) -> str:
    """Replace selected COCO object mentions with a neutral target-only token."""
    revised = str(caption)
    aliases: List[str] = []
    wanted = {str(obj).lower() for obj in canonical_objects}
    for phrase, canonical in _PHRASE2CANON.items():
        if canonical in wanted:
            aliases.append(phrase)
    for alias in sorted(set(aliases), key=lambda value: (len(value.split()), len(value)), reverse=True):
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in alias.split()) + r"(?!\w)"
        revised = re.sub(pattern, "item", revised, flags=re.I)
    revised = re.sub(r"\ba\s+item\b", "an item", revised, flags=re.I)
    return revised


def chair_ground_truth_objects(
    instance_objects: Sequence[str], references: Sequence[str]
) -> Set[str]:
    """Official CHAIR ground truth: segment objects union reference mentions."""
    gt = {_SYN2CANON.get(str(o).lower(), str(o).lower()) for o in instance_objects}
    for reference in references:
        gt.update(mentioned_objects(reference))
    return gt


def chair_i(caption: str, gt_objects: Sequence[str]) -> Dict[str, float]:
    """Instance-level CHAIR for a single caption."""
    gt = {_SYN2CANON.get(str(o).lower(), str(o).lower()) for o in gt_objects}
    mentioned = mentioned_object_instances(caption)
    if not mentioned:
        return {"n_mentioned": 0, "n_hallucinated": 0, "n_correct": 0}
    hallucinated = [obj for obj in mentioned if obj not in gt]
    return {
        "n_mentioned": len(mentioned),
        "n_hallucinated": len(hallucinated),
        "n_correct": len(mentioned) - len(hallucinated),
    }


def chair_i_corpus(captions: Sequence[str], gt_objects_list: Sequence[Sequence[str]]) -> float:
    tot_m, tot_h = 0, 0
    for cap, gt in zip(captions, gt_objects_list):
        r = chair_i(cap, gt)
        tot_m += r["n_mentioned"]
        tot_h += r["n_hallucinated"]
    return float(tot_h / tot_m) if tot_m else 0.0


def chair_i_reduction_ci(
    vanilla: Sequence[str],
    revised: Sequence[str],
    gt_objects_list: Sequence[Sequence[str]],
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired image-bootstrap CI for the absolute CHAIR-i reduction.

    Positive values mean fewer hallucinated object mentions after revision.
    The corpus numerator and denominator are recomputed for every bootstrap
    draw; image-level CHAIR ratios are not averaged.
    """
    if not (len(vanilla) == len(revised) == len(gt_objects_list)):
        raise ValueError("CHAIR paired inputs must have identical lengths")
    point = chair_i_corpus(vanilla, gt_objects_list) - chair_i_corpus(
        revised, gt_objects_list
    )
    if len(vanilla) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(int(n_boot)):
        indices = rng.integers(0, len(vanilla), size=len(vanilla))
        sampled_vanilla = [vanilla[int(index)] for index in indices]
        sampled_revised = [revised[int(index)] for index in indices]
        sampled_gt = [gt_objects_list[int(index)] for index in indices]
        values.append(
            chair_i_corpus(sampled_vanilla, sampled_gt)
            - chair_i_corpus(sampled_revised, sampled_gt)
        )
    return (
        float(point),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def object_recall(captions: Sequence[str], gt_objects_list: Sequence[Sequence[str]]) -> float:
    tot_gt, tot_hit = 0, 0
    for cap, gt in zip(captions, gt_objects_list):
        gt_set = {_SYN2CANON.get(str(o).lower(), str(o).lower()) for o in gt}
        if not gt_set:
            continue
        tot_gt += len(gt_set)
        tot_hit += len(gt_set & mentioned_objects(cap))
    return float(tot_hit / tot_gt) if tot_gt else 0.0


def bleu4(captions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smooth = SmoothingFunction().method1
    hyps = [str(c).lower().split() for c in captions]
    refs = [[str(r).lower().split() for r in refset] for refset in references]
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return 0.0
    hyps = [h for h, _ in pairs]
    refs = [r for _, r in pairs]
    return float(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))


def rouge_l(captions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals: List[float] = []
    for cap, refset in zip(captions, references):
        if not refset:
            continue
        vals.append(max(scorer.score(str(r), str(cap))["rougeL"].fmeasure for r in refset))
    return float(np.mean(vals)) if vals else 0.0


def summarise(
    captions: Sequence[str],
    gt_objects_list: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> Dict[str, float]:
    return {
        "CHAIR-i": chair_i_corpus(captions, gt_objects_list),
        "BLEU-4": bleu4(captions, references),
        "ROUGE-L": rouge_l(captions, references),
        "obj_recall": object_recall(captions, gt_objects_list),
    }
